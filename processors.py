import torch
from PIL import Image

from torchvision import transforms as T
from transformers import AutoModelForDepthEstimation

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
