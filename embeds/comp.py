import os
import sys

path = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(path)
os.chdir(path)

import time
import torch
import torch.nn as nn
import mobileclip
from torchvision.transforms import v2 as T
from torch.nn import functional as F
from PIL import Image

from processors import *

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

clip, preprocess = getCLIP()
enc = mobileclip.get_tokenizer('mobileclip_s0')

frame = torch.stack([
    preprocess(Image.open("./imgs/dogcat.jpg").convert('RGB')),
    # preprocess(Image.open("./imgs/modern.png").convert('RGB')),
    # preprocess(Image.open("./imgs/ghostgirl.png").convert('RGB'))
]).to(device)

words = open('./embeds/words.txt').read().split('\n')

img = clip.encode_image(frame, patch=True).to(device, dtype)
txt = torch.load('./embeds/words.emb').to(device, dtype)

ish = img.view(-1, 512, 64).permute(0, 2, 1).unsqueeze(2)
tsh = txt.unsqueeze(0).unsqueeze(1)

cos = F.cosine_similarity(ish, tsh, dim=-1)
wgh = F.softmax(cos / 0.001, dim=-1)

out = wgh @ tsh.squeeze(0).expand(wgh.size(0), -1, -1)
out = out.permute(0, 2, 1).view(out.size(0), 512, 8, 8)

# print(torch.cosine_similarity(img, out))

for y in range(8):
    for x in range(8):
        cmp = F.normalize(txt) @ F.normalize(out[:, :, y, x]).T
        print(words[cmp.argmax()], end=' ')
    print()

###################################

# cmp = F.normalize(txt) @ F.normalize(img).T
# out = F.softmax(cmp / 0.001, dim=0).T @ txt

# print(torch.cosine_similarity(img, out))

# cmp = F.normalize(txt) @ F.normalize(out).T

# for i in torch.topk(cmp, k=25, dim=0).indices:
#     print(words[i])
#     print(torch.cosine_similarity(out, txt[i]).item())

# print('"' + words[0] + '"')
# print(torch.cosine_similarity(out, txt[0]).item())

# for i in torch.topk(cmp.T, k=25, dim=0).values:
#     print(i)
