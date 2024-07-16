import numpy as np
import torch
from PIL import Image

from diffusers import AutoencoderKL, AutoencoderTiny
from genpercept.models import CustomUNet2DConditionModel
from genpercept.pipeline_genpercept import GenPerceptPipeline

class InversePipe:
    def __init__(self, mode):
        device = torch.device("cuda")
        dtype = torch.float16

        unet = CustomUNet2DConditionModel.from_pretrained(f'./models/{mode}', torch_dtype=dtype)
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=dtype)
        empty_text_embed = torch.from_numpy(np.load("./genpercept/empty_text_embed.npy")).to(device, dtype)[None] # [1, 77, 1024]

        genpercept_params_ckpt = dict(
            unet=unet,
            vae=vae,
            empty_text_embed=empty_text_embed,
            customized_head=None,
        )

        self.pipe = GenPerceptPipeline(**genpercept_params_ckpt)

        self.pipe = self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not loaded :(")
            pass  # run without xformers

        self.mode = mode
    
    def run(self, images):
        out = self.pipe.single_infer(images, mode=self.mode).clip(-1, 1)
        if self.mode == 'depth':
            out = (out + 1.0) * 0.5
        return out

from torchvision import transforms as T

depth = InversePipe('depth')
images = torch.stack([
    T.ToTensor()(Image.open('./imgs/ghostgirl.png').convert("RGB").resize((256, 256))).to(torch.float16).cuda(),
    T.ToTensor()(Image.open('./imgs/coffeegirl.jpg').convert("RGB").resize((256, 256))).to(torch.float16).cuda(),
    T.ToTensor()(Image.open('./imgs/dogcat.jpg').convert("RGB").resize((256, 256))).to(torch.float16).cuda(),
])

import time
t = time.time()
out = depth.run(images)
print(time.time() - t)
input()

# from genpercept.util.image_util import colorize_depth_maps, norm_to_rgb
# out.pred_colored.save('./out.png')
