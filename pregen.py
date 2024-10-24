import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb

from random import uniform as randf
from random import randint as randi
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from PIL import Image

from GFLIP import Generator
from datasets import ImageSet
from processors import *

wandb.require("core")
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
batchsize = 16
epochs = 10
loadpt = -1
stats = False

print("Loading Data... ", end = "")
dataset = ImageSet("C:/Datasets/Imagenet/Data")
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)
ld = len(dataloader)
print(ld * batchsize, "images loaded")

gen = Generator().to(device)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

paramsG = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

if stats:
    wandb.init(
        project = 'GFLIP-A',
        config = {
            'gen_params': paramsG,
            'dis_params': 0,
            'batchsize': batchsize,
        }
    )

print("Gen Params:", paramsG)

optimizerG = optim.NAdam(gen.parameters(), lr=0.0003, betas=(0.1, 0.9))
scalerG = torch.cuda.amp.GradScaler()

def mse(x, t):
    return (x - t).square().mean()

def noiselt(x, p):
    return x * (1 - p) + torch.randn_like(x) * p

depth = DepthPipe(518)
halve = nn.Upsample(scale_factor=0.5)
i2t = img2txt()
clip, preprocess = getCLIP()
vae = getVAE()

frameT = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
embedT = i2t(clip.encode_image(frameT, patch=True))
depthT = halve(depth(frameT))
ltimgT = vae.encode(frameT, False)[0]

t = time.time()

for epoch in range(epochs):
    for i, (frameA, frameB) in enumerate(dataloader):

        bs = frameA.size(0)

        frameA = frameA.to(device)
        frameB = frameB.to(device)

        depthA = depth(frameA)

        optimizerG.zero_grad()
        with torch.cuda.amp.autocast():

            ltimgA = vae.encode(frameA, False)[0]
            ltimgB = vae.encode(frameB, False)[0]
            embedA = clip.encode_image(frameA, patch=True)

            embedI = i2t(embedA)

            ltimgI = torch.stack([noiselt(ltimgB[x], x/(bs-1)) for x in range(bs)])

            ltimgC = gen(ltimgI, halve(depthA), embedI)

            frameC = vae.decode(ltimgC, return_dict=False)[0].clamp(0, 1)
            embedC = clip.encode_image(frameC, patch=True)
            clipsim = torch.cosine_similarity(embedA, embedC).mean()

            gloss = mse(ltimgA, ltimgC) + mse(depth(frameC), depthA) + (1 - clipsim) * 0.1

        scalerG.scale(gloss).backward()
        scalerG.step(optimizerG)
        scalerG.update()
        if scalerG.get_scale() < 64:
            scalerG.update(16384.0)
        nn.utils.clip_grad_norm_(gen.parameters(), 0.1)

        if stats:
            wandb.log({
                'gloss': gloss,
                'dloss': 0,
                'clip_siml': clipsim
            })
        
        if i % 50 == 0:

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{ld}] [loss: {gloss.item():.4f}] [CLIP: {clipsim.item():.4f}] [time: {(time.time() - t):.1f}s]")
            
            with torch.no_grad():
                tests = []

                for e in range(4):
                    test = vae.decode(gen(noiselt(ltimgT, e/3), depthT, embedT), return_dict=False)[0].clamp(0, 1)
                    tests.append(test)

                outimgs = T.ToPILImage()(make_grid(torch.cat(tests), 2))
                outimgs.save(f"./results/{epoch}-{i}.png")

                if stats:
                    wandb.log({
                        'outputs': wandb.Image(outimgs, caption=f'{epoch}-{i}')
                    })

                torch.save(gen, f"./models/gen_{epoch}.pth")
            
            t = time.time()

if stats:
    wandb.finish()