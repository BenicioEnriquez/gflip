import time
import torch
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.utils import make_grid
from PIL import Image
from pyperlin import FractalPerlin2D

import mobileclip
from GFLIP import Generator

torch.backends.cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Settings
loadpt = 0
mult = 1
dtype = torch.float16

clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s0', pretrained=f'./models/mobileclip_s0.pt')
clip = clip.to(device, dtype)

for p in clip.image_encoder.parameters():
    p.requires_grad = False
clip.image_encoder.eval()

for p in clip.text_encoder.parameters():
    p.requires_grad = False
clip.text_encoder.eval()

gen = Generator().to(device, dtype)

gen.load_state_dict(torch.load(f'./models/gen_{loadpt}.pth').state_dict())

params = np.sum([p.numel() for p in gen.parameters()]).item()/10**6

print("Params:", params)

scale = torch.nn.Upsample(scale_factor=mult, mode='bilinear')
test_noise = torch.zeros(1, 3, int(256 * mult), int(256 * mult)).to(device, dtype)
test_image = preprocess(Image.open("./imgs/boujee.png").convert('RGB')).unsqueeze(0).to(device, dtype)
test_embeds = clip.encode_image(test_image, patch=True)
test_image = scale(test_image)
test_embeds = scale(test_embeds)

images = [test_image]
img = test_noise

with torch.inference_mode():
    for x in range(8):
        t = time.time()
        img = gen(img, test_embeds)
        print('%.4fms -> ' % ((time.time()-t) * 1000), end='')
        images.append(img)
        guess_embeds = clip.encode_image(img, patch=True)
        clipsim = torch.cosine_similarity(guess_embeds, test_embeds).mean()
        print('%.4f' % clipsim.item())

T.ToPILImage()(make_grid(torch.concat(images), 3)).save(f"./out.png")
