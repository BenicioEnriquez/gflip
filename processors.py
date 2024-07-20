import numpy as np
import torch
from PIL import Image

from torchvision import transforms as T
from diffusers import AutoencoderTiny
from genpercept.models import CustomUNet2DConditionModel
from genpercept.pipeline_genpercept import GenPerceptPipeline
from genpercept.util.image_util import colorize_depth_maps, norm_to_rgb, chw2hwc

class InversePipe:
    def __init__(self, mode):
        self.device = torch.device("cuda")
        self.dtype = torch.float16

        unet = CustomUNet2DConditionModel.from_pretrained(f'./models/{mode}', torch_dtype=self.dtype)
        vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype)
        empty_text_embed = torch.from_numpy(np.load("./genpercept/empty_text_embed.npy")).to(self.device, self.dtype)[None] # [1, 77, 1024]

        genpercept_params_ckpt = dict(
            unet=unet,
            vae=vae,
            empty_text_embed=empty_text_embed,
            customized_head=None,
        )

        self.pipe = GenPerceptPipeline(**genpercept_params_ckpt)

        self.pipe = self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        try:
            import xformers
            self.pipe.enable_xformers_memory_efficient_attention()
        except:
            print("xformers not loaded :(")
            pass  # run without xformers

        self.mode = mode
    
    def __call__(self, images):
        t = images.dtype
        out = self.pipe.single_infer(images.to(dtype=self.dtype), mode=self.mode).clip(-1, 1)
        if self.mode == 'depth':
            out = (out + 1.0) * 0.5
        return out.to(dtype=t)

def depth2img(t):
    t = colorize_depth_maps(t.cpu(), 0, 1)[0]
    return T.ToPILImage()(t)

def norm2img(t):
    t = chw2hwc(t.cpu().numpy())
    t = norm_to_rgb(t)
    return T.ToPILImage()(t)


# TEST BENCH

if __name__ == "__main__":
    depth = InversePipe('depth')

    res = 512

    images = torch.stack([
        T.ToTensor()(Image.open('./imgs/ghostgirl.png').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
        T.ToTensor()(Image.open('./imgs/coffeegirl.jpg').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
        T.ToTensor()(Image.open('./imgs/dogcat.jpg').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
    ])

    import time
    total = 0

    for i in range(50):
        t = time.time()
        out = depth.run(images)
        total += time.time() - t

    print(total / 50)
