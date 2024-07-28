import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class XBlock(nn.Module):
    def __init__(self, ch, mid, f1, f2):
        super(XBlock, self).__init__()

        self.cx = f2 != 0

        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, f1, 1, (f1-1)//2, 1, ch, False),
            nn.GroupNorm(1, ch),
            nn.Conv2d(ch, mid, 1, 1, 0, 1, 1, False),
            nn.ReLU(inplace=True),
        )
        if self.cx:
            self.mid = nn.Sequential(
                nn.Conv2d(mid, mid, f2, 1, (f2-1)//2, 1, mid, False),
                nn.ReLU(inplace=True),
            )
        self.tail = nn.Conv2d(mid, ch, 1, 1, 0, 1, 1, False)

    def forward(self, x):
        y = self.head(x)
        if self.cx:
            y = self.mid(y)
        return self.tail(y) + x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.headblk = nn.Sequential(
            nn.Conv2d(16, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
        )

        self.clipblk = nn.Sequential(
            nn.Conv2d(512, 512, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.mainblk = nn.Sequential(
            XBlock(128, 512, 7, 5),
            nn.ReLU(inplace=True),
            XBlock(128, 512, 7, 5),
            nn.ReLU(inplace=True),
            XBlock(128, 512, 5, 3),
            nn.ReLU(inplace=True),
            XBlock(128, 512, 5, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 8, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, m, c):
        x = torch.concat([x * m, m], dim=1)
        x = self.headblk(x)
        c = self.clipblk(c)
        x = torch.concat([x, c], dim=1)
        return self.mainblk(x)

