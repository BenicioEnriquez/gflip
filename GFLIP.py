import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GBlock(nn.Module):
    def __init__(self, ch, mid, f1, f2):
        super(GBlock, self).__init__()

        self.cx = f2 != 0

        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, f1, 1, (f1-1)//2, 1, ch, False),
            nn.GroupNorm(1, ch),
            nn.Conv2d(ch, mid, 1, 1, 0, 1, 1, False),
            nn.ReLU6(),
        )
        if self.cx:
            self.mid = nn.Sequential(
                nn.Conv2d(mid, mid, f2, 1, (f2-1)//2, 1, mid, False),
                nn.ReLU6(),
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

        self.clipblk4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 5, 1, 2, bias=False),
            nn.ReLU6(),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.clipblk8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.clipblk16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.clipblk32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU6()
        )

        self.imgblk32 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU6()
        )
        self.imgblk16 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.ReLU6()
        )
        self.imgblk8 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.ReLU6()
        )
        self.imgblk4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.ReLU6()
        )

        self.encode32 = GBlock(32, 96, 5, 3)
        self.encode16 = GBlock(64, 128, 5, 3)
        self.encode8 = GBlock(128, 384, 5, 3)
        self.encode4 = GBlock(256, 768, 7, 3)

        self.down16 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.down8 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.down4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)

        self.decode4 = nn.Sequential(
            GBlock(256, 768, 7, 3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.decode8 = nn.Sequential(
            GBlock(128, 256, 5, 3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.decode16 = nn.Sequential(
            GBlock(64, 128, 5, 3),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU6()
        )
        self.decode32 = nn.Sequential(
            GBlock(32, 96, 5, 3),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, depth, clip):

        x = torch.concat([x, depth], dim=1)
        
        feat4 = self.clipblk4(clip)
        feat8 = self.clipblk8(feat4)
        feat16 = self.clipblk16(feat8)
        feat32 = self.clipblk32(feat16)

        x = self.imgblk32(x)
        feat32 = self.encode32(x + feat32)

        x = self.imgblk16(x)
        feat16 = self.encode16(self.down16(feat32) + x + feat16)

        x = self.imgblk8(x)
        feat8 = self.encode8(self.down8(feat16) + x + feat8)

        x = self.imgblk4(x)
        feat4 = self.encode4(self.down4(feat8) + x + feat4)

        x = self.decode4(feat4)
        x = self.decode8(x + feat8)
        x = self.decode16(x + feat16)
        x = self.decode32(x + feat32)
        
        return x