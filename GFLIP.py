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

class Genloss(nn.Module):
    def __init__(self):
        super(Genloss, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            XBlock(64, 196, 7, 5),
            nn.Conv2d(64, 64, 3, 2, 1),
            XBlock(64, 196, 7, 5),
            nn.Conv2d(64, 64, 3, 2, 1),
            XBlock(64, 196, 7, 5),
            nn.Conv2d(64, 64, 3, 2, 1),
            XBlock(64, 196, 7, 5),
            nn.Conv2d(64, 64, 3, 2, 1),
            XBlock(64, 196, 7, 5),
            nn.Conv2d(64, 1, 3, 2, 1)
        )
    
    def forward(self, x, depth):
        x = torch.concat([x, depth], dim=1)
        return self.net(x).mean(dim=[1,2,3])


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.clipblk4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(512, 512, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 5, 1, 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.clipblk8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.clipblk16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.clipblk32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.imgblk32 = nn.Conv2d(4, 32, 3, 1, 1)
        self.imgblk16 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.imgblk8 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.imgblk4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)

        self.encode32 = XBlock(32, 96, 7, 5)
        self.encode16 = XBlock(64, 192, 7, 5)
        self.encode8 = XBlock(128, 384, 7, 5)
        self.encode4 = XBlock(256, 768, 7, 5)

        self.down16 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.down8 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.down4 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)

        self.decode4 = nn.Sequential(
            XBlock(256, 768, 7, 5),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.decode8 = nn.Sequential(
            XBlock(128, 256, 7, 5),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.decode16 = nn.Sequential(
            XBlock(64, 196, 7, 5),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.decode32 = nn.Sequential(
            XBlock(32, 96, 7, 5),
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