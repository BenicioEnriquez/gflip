import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GBlock(nn.Module):
    def __init__(self, ch):
        super(GBlock, self).__init__()

        mid = ch * 4

        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 5, 1, 2, 1, ch, False),
            nn.LayerNorm(),
            nn.Conv2d(ch, mid, 1, 1, 0, 1, 1, False),
            nn.ReLU6(),
            nn.Conv2d(mid, mid, 3, 1, 1, 1, mid, False),
            nn.ReLU6(),
            nn.Conv2d(mid, ch, 1, 1, 0, 1, 1, False)
        )

    def forward(self, x):
        return self.net(x) + x
    
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.prevhead = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.cliphead = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, prev, clip):
        with torch.cuda.amp.autocast():
            x = torch.concat([
                self.prevhead(prev),
                self.cliphead(clip)
            ], dim=1)
            x = self.block1(x)
            x = self.tail(x)
        return x