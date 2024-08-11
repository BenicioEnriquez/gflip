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

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.headblk = nn.Sequential(
            nn.PixelUnshuffle(4)
        )

        self.clipblk = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.mainblk = nn.Sequential(
            XBlock(512, 2048, 15, 9),
            XBlock(512, 2048, 15, 9),
            XBlock(512, 2048, 15, 9),
            XBlock(512, 2048, 15, 9),
            nn.PixelShuffle(2),
            XBlock(128, 512, 9, 5),
            XBlock(128, 512, 9, 5),
            XBlock(128, 512, 9, 5),
            XBlock(128, 512, 9, 5),
            nn.PixelShuffle(2),
            XBlock(32, 128, 9, 5),
            XBlock(32, 128, 9, 5),
            XBlock(32, 128, 9, 5),
            XBlock(32, 128, 9, 5),
            nn.Conv2d(32, 8, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, m, c):
        x = torch.concat([x * m, m], dim=1)
        x = self.headblk(x)
        c = self.clipblk(c)
        x = torch.concat([x, c], dim=1)
        return self.mainblk(x)

