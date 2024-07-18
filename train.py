import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from PIL import Image
import time
import wandb

import mobileclip
from GFLIP import Generator
from datasets import ImageSet

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
batchsize = 16
epochs = 10
loadpt = -1
stats = True

clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s0', pretrained=f'./models/mobileclip_s0.pt')
clip = clip.to(device)

for p in clip.image_encoder.parameters():
    p.requires_grad = False
clip.image_encoder.eval()

for p in clip.text_encoder.parameters():
    p.requires_grad = False
clip.text_encoder.eval()

dataset = ImageSet("C:/Datasets/Imagenet/Data")
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

gen = Generator().to(device)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

if stats:
    wandb.init(
        project = 'GFLIP-U',
        config = {
            'params': params,
            'batchsize': batchsize,
        }
    )

print("Params:", params)

optimizer = optim.NAdam(gen.parameters(), lr=0.0001, betas=(0.0, 0.9))
scaler = torch.cuda.amp.GradScaler()
check = nn.HuberLoss()

test_noise = torch.zeros(1, 3, 256, 256).to(device)
test_image = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
test_embeds = clip.encode_image(test_image, patch=True)

def train(inp, out):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out_embeds = clip.encode_image(out, patch=True)
        guess = gen(inp, out_embeds)
        
        # membeds = torch.cat((rembeds[1:], rembeds[0:1]), dim=0).detach()
        guess_embeds = clip.encode_image(guess, patch=True)
        clipsim = torch.cosine_similarity(guess_embeds, out_embeds).mean()

        loss = check(guess, out) + (1 - clipsim) * 0.01

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < 64:
        scaler.update(16384.0)

    if stats:
        wandb.log({
            'loss': loss,
            'clip_siml': clipsim
        })

        if clipsim.item() > 0.9:
            wandb.alert(title='EUREKA!', text='CLIP embedding cosine similarity is above 90%!!!', wait_duration=3600)
    
    return guess.detach(), loss.detach(), clipsim.detach()

for epoch in range(epochs):
    for i, (prev, target) in enumerate(dataloader):

        bs = target.shape[0]
        prev = prev.to(device)
        target = target.to(device)
        
        pred, loss, clipsim = train(prev, target)
        train(pred, target)
        
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}] [CLIP: {clipsim.item()}]")
            img = gen(test_noise, test_embeds)
            images = [test_image, img]
            for x in range(7):
                img = gen(img, test_embeds)
                images.append(img)
            T.ToPILImage()(make_grid(torch.concat(images), 3)).save(f"./results/{epoch}-{i}.png")
            torch.save(gen, f"./models/gen_{epoch}.pth")
            t = time.time()

if stats:
    wandb.finish()