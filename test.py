import time
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from PIL import Image

from GFLIP import Generator
from processors import DepthPipe, getCLIP

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
loadpt = 0
mult = 2
dtype = torch.float16

gen = Generator().to(device, dtype)

gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

print("Params:", params)

nsize = int(256 * mult)
depth = DepthPipe(518)
clip, preprocess = getCLIP()
clip.to(device, dtype)

def getmask(b, c, m, p):
    s = nsize // (2 ** p)
    return nn.Upsample((nsize, nsize))(torch.repeat_interleave((torch.rand(b, 1, s, s) < m).float(), c, 1).round()).to(device, dtype)

scale = torch.nn.Upsample(scale_factor=mult, mode='bilinear')
frameT = preprocess(Image.open("./imgs/modern.png").convert('RGB')).unsqueeze(0).to(device, dtype)
embedT = clip.encode_image(frameT, patch=True)
depthT = scale(depth(frameT))
frameT = scale(frameT)
embedT = scale(embedT)

tests = []

istack = torch.concat([
    depthT,
    frameT,
    depthT,
    frameT
], dim=1)

mask = torch.concat([
    getmask(1, 1, 0.15, 0),
    getmask(1, 3, 0.15, 0),
    getmask(1, 1, 0.15, 0),
    getmask(1, 3, 0.0, 0),
], dim=1)
imask = (1 - mask)

with torch.inference_mode():
    for x in range(4):
            
        t = time.time()
        ostack = gen(istack, mask, embedT, embedT) * imask + istack * mask
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')

        tests.append(torch.repeat_interleave(ostack[:, :1], 3, 1))
        tests.append(ostack[:, 1:4])
        tests.append(torch.repeat_interleave(ostack[:, 4:5], 3, 1))
        tests.append(ostack[:, -3:])

        embedG = clip.encode_image(ostack[:, -3:], patch=True)
        clipsim = torch.cosine_similarity(embedG, embedT).mean()
        print('%.4f' % clipsim.item())

        istack = torch.concat([
            depthT,
            frameT,
            # ostack[:, 1:4],
            depthT,
            ostack[:, -3:]
        ], dim=1)
        
        mask = torch.concat([
            getmask(1, 1, 0.15, 0),
            getmask(1, 3, 0.15, 0),
            getmask(1, 1, 0.15, 0),
            getmask(1, 3, 0.15, 0),
        ], dim=1)
        imask = (1 - mask)

T.ToPILImage()(make_grid(torch.concat(tests), 4)).save(f"./out.png")
