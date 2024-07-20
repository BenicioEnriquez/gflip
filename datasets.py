import os
import os.path
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

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
            T.RandomResizedCrop(384),
            T.RandomHorizontalFlip()
        ])
        self.tail = T.Compose([
            T.RandomCrop(256),
            T.ToDtype(torch.float32, scale=True)
        ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        image = self.head(image)
        return self.tail(image), self.tail(image)