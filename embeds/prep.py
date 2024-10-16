import os
import sys

path = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(path)
os.chdir(path)

import torch
import mobileclip
from processors import *

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

print('loading CLIP...', end=' ', flush=True)
clip, preprocess = getCLIP()
enc = mobileclip.get_tokenizer('mobileclip_s0')
print('done')

print('tokenizing...', end=' ', flush=True)
words = enc(open('./embeds/words.txt').readlines()).to(device)
print('done')

print('encoding...', end=' ', flush=True)
txt = clip.encode_text(words).to(device, dtype)
print('done')

print('saving...', end=' ', flush=True)
torch.save(txt, './embeds/words.emb')
print('done')