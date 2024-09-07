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
batchsize = 4
epochs = 10
loadpt = -1
stats = False

dataset = ImageSet("C:/Datasets/Imagenet/Data")
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

gen = Generator().to(device)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

if stats:
    wandb.init(
        project = 'GFLIP-V',
        config = {
            'params': params,
            'batchsize': batchsize,
        }
    )

print("Params:", params)

optimizer = optim.NAdam(gen.parameters(), lr=0.0001, betas=(0.0, 0.9))
scaler = torch.cuda.amp.GradScaler()
check = nn.HuberLoss()

depth = DepthPipe(518)
clip, preprocess = getCLIP()
vae = getVAE()

pixshuf = nn.PixelUnshuffle(2)

def compd(d):
    return pixshuf(nn.functional.interpolate(d, 64))

def getmask(b, c, m, p):
    s = 32 // (2 ** p)
    return nn.Upsample((32, 32))(torch.repeat_interleave((torch.rand(b, 1, s, s) < m).float(), c, 1).round()).to(device)

frameT = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
embedT = clip.encode_image(frameT, patch=True)
depthT = compd(depth(frameT))
vqimgT = vae.encode(frameT, False)[0]

t = time.time()

for epoch in range(epochs):
    for i, (frameA, frameB) in enumerate(dataloader):

        bs = frameA.size(0)

        frameA = frameA.to(device)
        frameB = frameB.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            embedA = clip.encode_image(frameA, patch=True)
            embedB = clip.encode_image(frameB, patch=True)

            depthA = compd(depth(frameA))
            depthB = compd(depth(frameB))

            vqimgA = vae.encode(frameA, False)[0]
            vqimgB = vae.encode(frameB, False)[0]

            istack = torch.concat([
                depthA,
                vqimgA,
                depthB,
                vqimgB
            ], dim=1)

            mask = torch.concat([
                getmask(bs, 4, randf(0.1, 0.3), randi(0, 4)),
                torch.concat([
                    getmask(bs-2, 16, randf(0.1, 0.3), randi(0, 4)),
                    torch.zeros(1, 16, 32, 32).to(device),
                    torch.zeros(1, 16, 32, 32).to(device),
                ]),
                getmask(bs, 4, randf(0.1, 0.3), randi(0, 4)),
                torch.concat([
                    torch.zeros(1, 16, 32, 32).to(device),
                    getmask(bs-2, 16, randf(0.1, 0.3), randi(0, 4)),
                    torch.zeros(1, 16, 32, 32).to(device),
                ]),
            ], dim=1)

            imask = (1 - mask)

            ostack = gen(istack, mask, embedA, embedB)
            
            loss = check(istack * imask, ostack * imask)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < 64:
            scaler.update(16384.0)

        if stats:
            wandb.log({
                'loss': loss
            })
        
        if i % 50 == 0:

            delta = time.time() - t

            with torch.no_grad():

                # TEST 1

                istack = torch.concat([
                    depthT,
                    vqimgT,
                    depthT,
                    torch.zeros(1, 16, 32, 32).to(device)
                ], dim=1)

                mask = torch.concat([
                    torch.ones(1, 24, 32, 32).to(device),
                    torch.zeros(1, 16, 32, 32).to(device)
                ], dim=1)

                tests = [
                    frameT,
                    nn.functional.interpolate(depthT[:, :3], 256),
                    vae.decode(gen(istack, mask, embedT, embedT)[:, -16:], return_dict=False)[0].clamp(0, 1)
                ]

                # TEST 2

                mask = torch.concat([
                    getmask(1, 4, 0.3, 3),
                    getmask(1, 16, 0.3, 3),
                    getmask(1, 4, 0.3, 3),
                    torch.zeros(1, 16, 32, 32).to(device)
                ], dim=1)
                imask = (1 - mask)

                ostack = gen(istack, mask, embedT, embedT) * imask + istack * mask
                tests.append(vae.decode(ostack[:, 4:20], return_dict=False)[0].clamp(0, 1))
                tests.append(nn.functional.interpolate(ostack[:, 20:23], 256))
                tests.append(vae.decode(ostack[:, -16:], return_dict=False)[0].clamp(0, 1))

                # TEST 3
                
                istack = torch.concat([
                    depthT,
                    vqimgT,
                    depthT,
                    torch.zeros(1, 16, 32, 32).to(device)
                ], dim=1)

                mask = torch.concat([
                    torch.ones(1, 4, 32, 32).to(device),
                    torch.zeros(1, 16, 32, 32).to(device),
                    torch.ones(1, 4, 32, 32).to(device),
                    torch.zeros(1, 16, 32, 32).to(device)
                ], dim=1)
                imask = (1 - mask)

                for m in range(3):
                    istack = gen(istack, mask, embedT, embedT) * imask + istack * mask
                    tests.append(vae.decode(ostack[:, -16:], return_dict=False)[0].clamp(0, 1))
                    mask = torch.concat([
                        torch.ones(1, 4, 32, 32).to(device),
                        torch.zeros(1, 16, 32, 32).to(device),
                        torch.ones(1, 4, 32, 32).to(device),
                        getmask(1, 16, 0.2, 3-m)
                    ], dim=1)
                    imask = (1 - mask)

                clipsim = torch.cosine_similarity(clip.encode_image(tests[-1], patch=True), embedT).mean()

                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item():.8f}] [CLIP: {clipsim.item():.4f}] [time: {delta:.1f}s]")

                outimgs = T.ToPILImage()(make_grid(torch.concat(tests), 3))
                outimgs.save(f"./results/{epoch}-{i}.png")
                if stats:
                    wandb.log({
                        'outputs': wandb.Image(outimgs.resize((512, 512)), caption=f'{epoch}-{i}'),
                        'clipsim': clipsim
                    })
                torch.save(gen, f"./models/gen_{epoch}.pth")
            
            t = time.time()

if stats:
    wandb.finish()