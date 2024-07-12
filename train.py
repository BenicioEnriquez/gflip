import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from PIL import Image
import time
import wandb

import mobileclip
from GFLIP import Generator
from datasets import ImageSet

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# Settings
batchsize = 12
epochs = 100
loadpt = 2

clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s0', pretrained=f'./models/mobileclip_s0.pt')
clip = clip.to(device, dtype=dtype)

for p in clip.image_encoder.parameters():
    p.requires_grad = False
clip.image_encoder.eval()

for p in clip.text_encoder.parameters():
    p.requires_grad = False
clip.text_encoder.eval()

tform = T.Compose([
    T.ToImage(),
    T.RandomResizedCrop(256),
    T.RandomHorizontalFlip(),
    T.ToDtype(dtype, scale=True)
])

dataset = ImageSet("C:/Datasets/Imagenet/Data", tform)
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

gen = Generator().to(device, dtype=dtype)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

wandb.init(
    project = 'GFLIP',
    config = {
        'params': params,
        'batchsize': batchsize,
    }
)

print("Params:", params)

optimizer = optim.NAdam(gen.parameters(), lr=0.0001, betas=(0.0, 0.9))
scaler = torch.cuda.amp.GradScaler()

tembed = clip.encode_image(preprocess(Image.open("./dogcat.jpg").convert('RGB')).cuda().unsqueeze(0))

for epoch in range(epochs):
    for i, images in enumerate(dataloader):

        bs = images.shape[0]
        images = images.to(device, dtype=dtype)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            rembeds = clip.encode_image(images)

            membeds = torch.cat((rembeds[1:], rembeds[0:1]), dim=0).detach()

            clipsim = torch.cosine_similarity(fembeds, rembeds).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < 64:
            scaler.update(16384.0)

        wandb.log({
            'real_loss': rloss,
            'fake_loss': floss,
            'mism_loss': mloss,
            'magp_loss': ploss,
            'netg_loss': gloss,
            'netd_loss': dloss,
            'genr_loss': -closs.mean(),
            'clip_siml': clipsim
        })

        if clipsim.item() > 0.9:
            wandb.alert(title='EUREKA!', text='CLIP embedding cosine similarity is above 90%!!!', wait_duration=3600)
        
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [G loss: {gloss.item()}] [CLIP: {clipsim.item()}]")
            T.ToPILImage()(gen(tnoise, tembed)[0]).save(f"./results/{epoch}-{i}.png")
            torch.save(gen, f"./models/gen_{epoch}.pth")
            t = time.time()

wandb.finish()