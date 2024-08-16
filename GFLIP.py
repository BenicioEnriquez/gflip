import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class XBlock(nn.Module):
    def __init__(self, ch, mid, f1, f2):
        super(XBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, f1, 1, (f1-1)//2, 1, ch, False),
            nn.GroupNorm(1, ch),
            nn.Conv2d(ch, mid, 1, 1, 0, 1, 1, False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, f2, 1, (f2-1)//2, 1, mid, False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, 1, 0, 1, 1, False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x) + x

class YBlock(nn.Module):
    def __init__(self, i, o):
        super(YBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.headblk = nn.Sequential(
            nn.PixelUnshuffle(8)
        )

        self.clipblk = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            YBlock(512, 396),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            YBlock(396, 256),
        )

        self.mainblk = nn.Sequential(
            XBlock(1536, 1536 * 3, 7, 5),
            YBlock(1536, 1536),
            XBlock(1536, 1536 * 3, 7, 5),
            YBlock(1536, 2048),
            XBlock(2048, 2048 * 3, 7, 5),
            YBlock(2048, 2048),

            nn.PixelShuffle(2),

            XBlock(512, 512 * 3, 7, 5),
            YBlock(512, 768),
            XBlock(768, 768 * 3, 7, 5),
            YBlock(768, 1024),

            nn.PixelShuffle(2),

            XBlock(256, 256 * 3, 5, 3),
            YBlock(256, 396),
            XBlock(396, 396 * 3, 5, 3),
            YBlock(396, 512),
            
            nn.PixelShuffle(2),

            XBlock(128, 128 * 3, 5, 3),
            YBlock(128, 128),

            nn.Conv2d(128, 8, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, m, c1, c2):
        x = torch.concat([x * m, m], dim=1)
        x = self.headblk(x)
        c1 = self.clipblk(c1)
        c2 = self.clipblk(c2)
        x = torch.concat([x, c1, c2], dim=1)
        return self.mainblk(x)

