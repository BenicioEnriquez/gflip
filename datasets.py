import os
import os.path
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from pyperlin import FractalPerlin2D

class ImageSet(Dataset):
    def __init__(self, dir):
        self.paths = [
            os.path.join(root, name)
            for root, dirs, files in os.walk(dir)
            for name in files
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.head = T.Compose([
            T.ToImage(),
            T.RandomResizedCrop(256),
            T.RandomHorizontalFlip()
        ])
        self.mod = T.Compose([
            T.GaussianBlur(kernel_size=(9, 9), sigma=(5, 10))
        ])
        self.tail = T.Compose([
            T.ToDtype(torch.float32, scale=True)
        ])
        resolutions = [(2**i,2**i) for i in range(1,7)]
        factors = [.5**i for i in range(6)]
        self.perlin = FractalPerlin2D((3, 256, 256), resolutions, factors)
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.head(image)
        prev = self.mod(image)

        if torch.rand(1) < 0.1:
            prev = self.mod(self.perlin())

        if torch.rand(1) < 0.1:
            prev = torch.zeros(3, 256, 256)

        return self.tail(prev), self.tail(image)