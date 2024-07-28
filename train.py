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
from PIL import Image

import mobileclip
from GFLIP import Generator
from datasets import ImageSet
from processors import DepthPipe

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
batchsize = 8
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
        project = 'GFLIP-M',
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
dblur = T.GaussianBlur((9, 7), (10, 20))

frameT = preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')).cuda().unsqueeze(0)
embedT = clip.encode_image(frameT, patch=True)
depthT = dblur(depth(frameT))

def getmask(b, c, p):
    return torch.repeat_interleave(torch.rand((b, 1, 256, 256)) * (0.5 + p * 0.5), c, 1).round().to(device)

t = time.time()

for epoch in range(epochs):
    for i, (frameA, frameB) in enumerate(dataloader):

        frameA = frameA.to(device)
        frameB = frameB.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            embedB = clip.encode_image(frameB, patch=True)

            depthA = dblur(depth(frameA))
            depthB = dblur(depth(frameB))

            istack = torch.concat([
                depthA,
                frameA,
                depthB,
                frameB
            ], dim=1)

            mask = torch.concat([
                getmask(istack.size(0), 1, 0.2),
                getmask(istack.size(0), 3, 0.2),
                getmask(istack.size(0), 1, 0.2),
                getmask(istack.size(0), 3, 0.2)
            ], dim=1)

            ostack = gen(istack, mask, embedB)
            
            frameC = ostack[:, -3:]

            embedC = clip.encode_image(frameC, patch=True)
            clipsim = torch.cosine_similarity(embedB, embedC).mean()

            imask = (1 - mask)
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

                istack = torch.concat([
                    depthT,
                    frameT,
                    depthT,
                    torch.zeros(1, 3, 256, 256).to(device)
                ], dim=1)

                mask = torch.concat([
                    torch.ones(1, 5, 256, 256).to(device),
                    torch.zeros(1, 3, 256, 256).to(device)
                ], dim=1)

                tests = [
                    frameT,
                    torch.repeat_interleave(depthT, 3, 1),
                    gen(istack, mask, embedT)[:, -3:]
                ]

                mask = torch.concat([
                    getmask(1, 1, 0.2),
                    getmask(1, 3, 0.2),
                    getmask(1, 1, 0.2),
                    torch.zeros(1, 3, 256, 256).to(device)
                ], dim=1)

                ostack = gen(istack, mask, embedT)
                tests.append(ostack[:, 1:4])
                tests.append(torch.repeat_interleave(ostack[:, :1], 3, 1))
                tests.append(ostack[:, -3:])
                
                mask = torch.concat([
                    torch.ones(1, 1, 256, 256).to(device),
                    torch.zeros(1, 3, 256, 256).to(device),
                    torch.ones(1, 1, 256, 256).to(device),
                    torch.zeros(1, 3, 256, 256).to(device)
                ], dim=1)

                for _ in range(3):
                    istack = gen(istack, mask, embedT)
                    tests.append(istack[:, -3:])
                    mask = torch.concat([
                        getmask(istack.size(0), 1, 0.2),
                        getmask(istack.size(0), 3, 0.2),
                        getmask(istack.size(0), 1, 0.2),
                        getmask(istack.size(0), 3, 0.2)
                    ], dim=1)

                T.ToPILImage()(make_grid(torch.concat(tests), 3)).save(f"./results/{epoch}-{i}.png")
                torch.save(gen, f"./models/gen_{epoch}.pth")
            
            t = time.time()

if stats:
    wandb.finish()