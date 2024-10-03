import torch
import torch.nn as nn
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
        )

    def forward(self, x):
        return self.net(x) + x

class YBlock(nn.Module):
    def __init__(self, i, o):
        super(YBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, 1, 1)
        )

        self.skip = nn.Conv2d(i, o, 1, 1, 0, 1, 1, False) if i != o else nn.Identity()
        self.fuse = nn.ReLU()
    
    def forward(self, x):
        return self.fuse(self.net(x) + self.skip(x))

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x) * 3

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            YBlock(512, 1024),
            XBlock(1024, 1024 * 4, 3, 1),
            YBlock(1024, 1024),
            XBlock(1024, 1024 * 4, 3, 1),

            nn.PixelShuffle(2),

            YBlock(256, 512),
            XBlock(512, 512 * 4, 3, 1),
            YBlock(512, 512),
            XBlock(512, 512 * 4, 3, 1),
            YBlock(512, 512),
            XBlock(512, 512 * 4, 3, 1),
            
            nn.PixelShuffle(2),

            YBlock(128, 256),
            XBlock(256, 256 * 4, 3, 1),
            YBlock(256, 256),
            XBlock(256, 256 * 4, 3, 1),
            YBlock(256, 256),
            XBlock(256, 256 * 4, 3, 1),
            YBlock(256, 256),
            XBlock(256, 256 * 4, 3, 1),

            nn.Conv2d(256, 16, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(16, 512, 3, 1, 1),

            XBlock(512, 512 * 4, 3, 1),
            nn.Conv2d(512, 128, 3, 1, 1),

            nn.PixelUnshuffle(2),

            XBlock(512, 512 * 4, 3, 1),
            nn.Conv2d(512, 128, 3, 1, 1),
            
            nn.PixelUnshuffle(2),

            XBlock(512, 512 * 4, 3, 1),
            nn.Conv2d(512, 128, 3, 1, 1),

            nn.Conv2d(128, 1, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)