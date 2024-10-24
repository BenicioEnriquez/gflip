import time
import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from PIL import Image

from GFLIP import Generator
from processors import *

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

i2t = img2txt()
depth = DepthPipe(518)
halve = nn.Upsample(scale_factor=0.5)
vae = getVAE().to(device, dtype)
clip, preprocess = getCLIP()
clip.to(device, dtype)

scale = nn.Upsample(scale_factor=mult, mode='bilinear')
frameT = preprocess(Image.open("./imgs/modern.png").convert('RGB')).unsqueeze(0).to(device, dtype)
frameA = nn.Upsample(256)(T.RandomResizedCrop(180)(frameT))
frameB = nn.Upsample(256)(T.RandomResizedCrop(180)(frameT))
embedI = clip.encode_image(frameB, patch=True)
embedT = i2t(embedI)
depthT = scale(halve(depth(frameB)))
frameA = scale(frameA)
frameB = scale(frameB)
embedI = scale(embedI)
embedT = scale(embedT)

ltimgA = vae.encode(frameA, False)[0]

def noiselt(x, p):
    return x * (1 - p) + torch.randn_like(x) * p

mask = torch.ones_like(frameA)
for x in range(mask.size(2)):
    for y in range(mask.size(3)):
        if x % 32 == 0 or y % 32 == 0:
            mask[0, 0, x, y] = 0
            mask[0, 1, x, y] = 0
            mask[0, 2, x, y] = 0

tests = [frameA * mask]

ltimgG = ltimgA

with torch.inference_mode():
    for x in range(8):
            
        t = time.time()
        ltimgG = gen(noiselt(ltimgA, x/7), depthT, embedT)
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')

        t = time.time()
        frameG = vae.decode(ltimgG, return_dict=False)[0].clamp(0, 1)
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')

        embedG = clip.encode_image(frameG, patch=True)
        clipsim = torch.cosine_similarity(embedG, embedI).mean()
        print('%.4f' % clipsim.item())

        tests.append(frameG * mask)

T.ToPILImage()(make_grid(torch.cat(tests), 3)).save(f"./out.png")
