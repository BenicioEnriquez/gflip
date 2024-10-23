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

from GFLIP import Generator, Discriminator
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
dis = Discriminator().to(device)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())
    dis.load_state_dict(torch.load(f'./models/dis_{loadpt}.pth').state_dict())

paramsG = np.sum([p.numel() for p in gen.parameters()]).item()/10**6
paramsD = np.sum([p.numel() for p in dis.parameters()]).item()/10**6

if stats:
    wandb.init(
        project = 'GFLIP-A',
        config = {
            'gen_params': paramsG,
            'dis_params': paramsD,
            'batchsize': batchsize,
        }
    )

print("Gen Params:", paramsG)
print("Dis Params:", paramsD)

optimizerG = optim.NAdam(gen.parameters(), lr=0.0001, betas=(0.1, 0.9))
optimizerD = optim.NAdam(dis.parameters(), lr=0.0001, betas=(0.5, 0.9))
scalerG = torch.cuda.amp.GradScaler()
scalerD = torch.cuda.amp.GradScaler()

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
            embedB = clip.encode_image(frameB, patch=True)

            embedI = i2t(embedA)

            ltimgI = torch.stack([noiselt(ltimgB[x], x/(bs-1)) for x in range(bs)])

            ltimgC = gen(ltimgI, halve(depthA), embedI)

            frameC = vae.decode(ltimgC, return_dict=False)[0].clamp(0, 1)
            embedC = clip.encode_image(frameC, patch=True)
            clipsim = torch.cosine_similarity(embedA, embedC).mean()

            fake = dis(ltimgC, embedC, ltimgI, embedB)
            real = dis(ltimgA, embedA, ltimgI, embedB)

            # gloss = (torch.mean(torch.nn.ReLU()(1.0 + (real - torch.mean(fake)))) + torch.mean(torch.nn.ReLU()(1.0 - (fake - torch.mean(real)))))/2
            gloss = (torch.mean((real - torch.mean(fake) + 1) ** 2) + torch.mean((fake - torch.mean(real) - 1) ** 2))/2
            tloss = mse(ltimgA, ltimgC) * 0.5 + mse(depth(frameC), depthA) + (1 - clipsim) * 0.1

        scalerG.scale(gloss + tloss * 0.5).backward()
        scalerG.step(optimizerG)
        scalerG.update()
        if scalerG.get_scale() < 64:
            scalerG.update(16384.0)
        nn.utils.clip_grad_norm_(gen.parameters(), 0.1)

        optimizerD.zero_grad()
        with torch.cuda.amp.autocast():

            fake = dis(ltimgC.detach(), embedC.detach(), ltimgI, embedB)
            real = dis(ltimgA, embedA, ltimgI, embedB)

            # dloss = (torch.mean(torch.nn.ReLU()(1.0 - (real - torch.mean(fake)))) + torch.mean(torch.nn.ReLU()(1.0 + (fake - torch.mean(real)))))/2
            dloss = (torch.mean((real - torch.mean(fake) - 1) ** 2) + torch.mean((fake - torch.mean(real) + 1) ** 2))/2

        scalerD.scale(dloss).backward()
        scalerD.step(optimizerD)
        scalerD.update()
        if scalerD.get_scale() < 64:
            scalerD.update(16384.0)
        nn.utils.clip_grad_norm_(dis.parameters(), 0.1)

        if stats:
            wandb.log({
                'gloss': gloss,
                'dloss': dloss,
                'clip_siml': clipsim
            })
        
        if i % 50 == 0:

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{ld}] [loss: {gloss.item():.4f}/{dloss.item():.4f}] [CLIP: {clipsim.item():.4f}] [time: {(time.time() - t):.1f}s]")
            
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
                torch.save(dis, f"./models/dis_{epoch}.pth")
            
            t = time.time()

if stats:
    wandb.finish()