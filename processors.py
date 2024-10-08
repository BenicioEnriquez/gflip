import torch
import mobileclip

from PIL import Image
from torchvision import transforms as T
from transformers import AutoModelForDepthEstimation
from diffusers import AutoencoderTiny

def getCLIP():
    clip, _, preprocess = mobileclip.create_model_and_transforms(f'mobileclip_s0', pretrained=f'./models/mobileclip_s0.pt')
    clip = clip.to(torch.device("cuda"))

    for p in clip.image_encoder.parameters():
        p.requires_grad = False
    clip.image_encoder.eval()

    for p in clip.text_encoder.parameters():
        p.requires_grad = False
    clip.text_encoder.eval()

    return clip, preprocess

def getVAE():
    vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3")
    vae = vae.to(torch.device("cuda"))

    for p in vae.encoder.parameters():
        p.requires_grad = False
    vae.encoder.eval()

    for p in vae.decoder.parameters():
        p.requires_grad = False
    vae.decoder.eval()

    return vae

class DepthPipe:
    def __init__(self, size=None):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.size = size

        self.model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf', torch_dtype=self.dtype).to(self.device)

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    
    def __call__(self, images):

        if self.size != None:
            b, c, h, w = images.size()
            short = min(h, w)
            s = self.size / short

            images = torch.nn.functional.interpolate(
                images,
                size=(int(h*s), int(w*s)),
                mode="bilinear"
            )
        
        with torch.no_grad():
            outputs = self.model(images)
            depth = outputs.predicted_depth

        depth = (depth / depth.max()).clamp(0, 1).unsqueeze(1)

        if self.size != None:
            depth = torch.nn.functional.interpolate(
                depth,
                size=(h, w),
                mode="bilinear"
            )

        return depth

# TEST BENCH

if __name__ == "__main__":
    depth = DepthPipe()

    res = 512

    imgs = torch.stack([
        T.ToTensor()(Image.open('./imgs/ghostgirl.png').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
        T.ToTensor()(Image.open('./imgs/coffeegirl.jpg').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
        T.ToTensor()(Image.open('./imgs/dogcat.jpg').convert("RGB").resize((res, res))).to(torch.float16).cuda(),
    ])

    import time
    total = 0

    for i in range(50):
        t = time.time()
        out = depth(imgs)
        total += time.time() - t

    print(total / 50)

    T.ToPILImage()(out[0]).show()
