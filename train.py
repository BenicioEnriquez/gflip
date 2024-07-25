import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb

from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from pytorch_msssim import MS_SSIM
from PIL import Image

import mobileclip
from GFLIP import Generator, Genloss
from datasets import ImageSet
from processors import InversePipe

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
batchsize = 8
epochs = 10
loadpt = 0
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
        project = 'GFLIP-D',
        config = {
            'params': params,
            'batchsize': batchsize,
        }
    )

print("Params:", params)

genloss = Genloss().to(device)

optimizer = optim.NAdam(gen.parameters(), lr=0.0001, betas=(0.0, 0.9))
scaler = torch.cuda.amp.GradScaler()
check = nn.HuberLoss()

l_optimizer = optim.NAdam(genloss.parameters(), lr=0.0002, betas=(0.0, 0.9))
l_scaler = torch.cuda.amp.GradScaler()
l_check = nn.BCEWithLogitsLoss()

ssim = MS_SSIM(1.0)
depth = InversePipe('depth')
dblur = T.GaussianBlur((9, 7), (10, 20))

test_noise = torch.zeros(1, 3, 256, 256).to(device)
test_image = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
test_embeds = clip.encode_image(test_image, patch=True)
test_depth = depth(test_image)
test_blurd = dblur(test_depth)

blank = torch.zeros(batchsize//2, 3, 256, 256).to(device)
ones = torch.ones(batchsize, device=device)
zeros = torch.zeros(batchsize, device=device)

def train(inp, dep, out):
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        out_embeds = clip.encode_image(out, patch=True)

        indep = dblur(dep)

        for _ in range(batchsize//2):
            indep[torch.randint(0, batchsize, (1,)).item()] = torch.zeros(1, 256, 256).to(device)

        guess = gen(inp, indep, out_embeds)
        
        guess_embeds = clip.encode_image(guess, patch=True)
        clipsim = torch.cosine_similarity(guess_embeds, out_embeds).mean()
        guess_depth = depth(guess)

        imgloss = check(guess, out)
        cliploss = (1 - clipsim)
        ssimloss = (1 - ssim(guess, out))
        depthloss = check(guess_depth, dep)
        dloss = F.sigmoid(genloss(guess, guess_depth)).mean()

        loss = cliploss * 0.25 + depthloss + imgloss + dloss * 0.1

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < 64:
        scaler.update(16384.0)

    l_optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        bad = genloss(guess.detach(), guess_depth.detach())
        good = genloss(out, dep)
        l_loss = l_check(bad, ones) + l_check(good, zeros)

    l_scaler.scale(l_loss).backward()
    l_scaler.step(l_optimizer)
    l_scaler.update()
    if l_scaler.get_scale() < 64:
        l_scaler.update(16384.0)

    if stats:
        wandb.log({
            'loss': loss,
            'g_loss': dloss,
            'd_loss': l_loss,
            'image_loss': imgloss,
            'ssim_loss': ssimloss,
            'depth_loss': depthloss,
            'clip_siml': clipsim
        })
    
    return guess.detach(), loss.detach(), clipsim.detach()

for epoch in range(epochs):
    for i, (prev, target) in enumerate(dataloader):

        prev = prev.to(device)
        target = target.to(device)

        tdepth = depth(target)
        pred, loss, clipsim = train(prev, tdepth, target)

        tdepth = depth(prev)
        train(torch.concat([pred[:batchsize//2], blank]), tdepth, prev)
        
        if i % 50 == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item()}] [CLIP: {clipsim.item()}]")
            img = test_noise.clone()
            images = [
                test_image,
                gen(img, torch.zeros(1, 1, 256, 256).to(device), test_embeds),
                gen(img, test_depth, test_embeds)
            ]
            for x in range(6):
                img = gen(img, test_blurd, test_embeds)
                images.append(img)
            T.ToPILImage()(make_grid(torch.concat(images), 3)).save(f"./results/{epoch}-{i}.png")
            torch.save(gen, f"./models/gen_{epoch}.pth")
            t = time.time()

if stats:
    wandb.finish()