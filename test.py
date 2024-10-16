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
mult = 1
dtype = torch.float16

gen = Generator().to(device, dtype)

gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

print("Params:", params)

depth = DepthPipe(518)
vae = getVAE().to(device, dtype)
clip, preprocess = getCLIP()
clip.to(device, dtype)

scale = torch.nn.Upsample(scale_factor=mult, mode='bilinear')
frameT = preprocess(Image.open("./imgs/modern.png").convert('RGB')).unsqueeze(0).to(device, dtype)
embedT = clip.encode_image(frameT, patch=True)
depthT = scale(depth(frameT))
frameT = scale(frameT)
embedT = scale(embedT)

ltimgT = vae.encode(frameT, False)[0]

tests = [frameT, torch.repeat_interleave(depthT, 3, 1)]

with torch.inference_mode():
    for x in range(8):
            
        t = time.time()
        ltimgG = gen(embedT)
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')

        t = time.time()
        frameG = vae.decode(ltimgG, return_dict=False)[0].clamp(0, 1)
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')

        embedG = clip.encode_image(frameG, patch=True)
        clipsim = torch.cosine_similarity(embedG, embedT).mean()
        print('%.4f' % clipsim.item())

tests.append(frameG)
tests.append(torch.repeat_interleave(depth(frameG), 3, 1))

T.ToPILImage()(make_grid(torch.concat(tests), 2)).save(f"./out.png")
