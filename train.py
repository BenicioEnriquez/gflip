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
from processors import DepthPipe, getCLIP

wandb.require("core")
torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
batchsize = 8
epochs = 10
loadpt = -1
stats = True

dataset = ImageSet("C:/Datasets/Imagenet/Data")
dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, pin_memory=True)

gen = Generator().to(device)

if loadpt > -1:
    gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

if stats:
    wandb.init(
        project = 'GFLIP-S',
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

frameT = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
embedT = clip.encode_image(frameT, patch=True)
depthT = depth(frameT)

def getprobs(b, c, r, p):
    s = 256 // (2 ** p)
    x = ((torch.rand(b, 1, s, s).to(device) * 2) - 1) + ((r * 2) - 1)
    x = nn.functional.sigmoid(x)
    x = torch.repeat_interleave(x, c, 1).round()
    return nn.Upsample((256, 256))(x)

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

            depthA = depth(frameA)
            depthB = depth(frameB)

            istack = torch.concat([
                depthA,
                frameA,
                depthB,
                frameB
            ], dim=1)

            probs = torch.concat([
                getprobs(bs, 1, randf(0.1, 0.3), randi(0, 4)),
                torch.concat([
                    getprobs(bs-2, 3, randf(0.1, 0.3), randi(0, 4)),
                    getprobs(2, 3, 0, 0),
                ]),
                getprobs(bs, 1, randf(0.1, 0.3), randi(0, 4)),
                torch.concat([
                    getprobs(1, 3, 0, 0),
                    getprobs(bs-2, 3, randf(0.1, 0.3), randi(0, 4)),
                    getprobs(1, 3, 0, 0),
                ]),
            ], dim=1)

            mask = (probs > 0.5).float()
            imask = (1 - mask)

            # T.ToPILImage()((istack * mask)[1, -3:]).show()

            noise = torch.randn(bs, 8, 256, 256).to(device)
            ostack = gen(istack, probs, embedA, embedB, noise)
            
            frameC = (istack * mask + ostack * imask)[:, -3:]

            embedC = clip.encode_image(frameC, patch=True)
            clipsim = torch.cosine_similarity(embedB, embedC).mean()
            
            loss = check(istack * imask, ostack * imask)

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
        
        if i % 50 == 0:

            print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [loss: {loss.item():.8f}] [CLIP: {clipsim.item():.4f}] [time: {(time.time() - t):.1f}s]")
            
            with torch.no_grad():

                noise = torch.randn(1, 8, 256, 256).to(device)

                # TEST 1

                istack = torch.concat([
                    depthT,
                    frameT,
                    depthT,
                    getprobs(1, 3, 0, 0),
                ], dim=1)

                probs = torch.concat([
                    getprobs(1, 5, 1, 0),
                    getprobs(1, 3, 0, 0),
                ], dim=1)

                tests = [
                    frameT,
                    torch.repeat_interleave(depthT, 3, 1),
                    gen(istack, probs, embedT, embedT, noise)[:, -3:]
                ]

                # TEST 2

                probs = torch.concat([
                    getprobs(1, 1, 0.3, 3),
                    getprobs(1, 3, 0.3, 3),
                    getprobs(1, 1, 0.3, 3),
                    getprobs(1, 3, 0, 0),
                ], dim=1)

                mask = (probs > 0.5).float()
                imask = (1 - mask)

                ostack = gen(istack, probs, embedT, embedT, noise) * imask + istack * mask
                tests.append(ostack[:, 1:4])
                tests.append(torch.repeat_interleave(ostack[:, 4:5], 3, 1))
                tests.append(ostack[:, -3:])

                # TEST 3
                
                istack = torch.concat([
                    depthT,
                    frameT,
                    depthT,
                    getprobs(1, 3, 0, 0),
                ], dim=1)

                probs = torch.concat([
                    getprobs(1, 1, 1, 0),
                    getprobs(1, 3, 0, 0),
                    getprobs(1, 1, 1, 0),
                    getprobs(1, 3, 0, 0),
                ], dim=1)

                mask = (probs > 0.5).float()
                imask = (1 - mask)

                for m in range(3):
                    istack = gen(istack, probs, embedT, embedT, noise) * imask + istack * mask
                    tests.append(istack[:, -3:])

                    probs = torch.concat([
                        getprobs(1, 1, 1, 0),
                        getprobs(1, 3, 0.2, 3-m),
                        getprobs(1, 1, 1, 0),
                        getprobs(1, 3, 0.2, 3-m)
                    ], dim=1)

                    mask = (probs > 0.5).float()
                    imask = (1 - mask)

                    noise = torch.randn(1, 8, 256, 256).to(device)

                outimgs = T.ToPILImage()(make_grid(torch.concat(tests), 3))
                outimgs.save(f"./results/{epoch}-{i}.png")
                if stats:
                    wandb.log({
                        'outputs': wandb.Image(outimgs.resize((512, 512)), caption=f'{epoch}-{i}')
                    })
                torch.save(gen, f"./models/gen_{epoch}.pth")
            
            t = time.time()

if stats:
    wandb.finish()