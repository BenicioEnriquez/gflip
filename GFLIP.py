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
        self.fuse = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.fuse(self.net(x) + self.skip(x))

class ZBlock(nn.Module):
    def __init__(self, i, o):
        super(ZBlock, self).__init__()

        self.net = nn.Conv2d(i, o, 3, 1, 1)
        self.skip = nn.Conv2d(i, o, 1, 1, 0, 1, 1, False) if i != o else nn.Identity()
        self.fuse = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.fuse(self.net(x) + self.skip(x))

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.dnet = nn.PixelUnshuffle(16)

        self.con1 = nn.Sequential(
            nn.PixelShuffle(2),
            YBlock(192, 256),
        )

        self.con2 = nn.Sequential(
            nn.PixelShuffle(4),
            YBlock(48, 128),
        )

        self.stg1 = nn.Sequential(
            XBlock(16, 16 * 3, 5, 3),
            nn.PixelUnshuffle(2),
            XBlock(64, 64 * 3, 5, 3),
            nn.PixelUnshuffle(2),
            XBlock(256, 256 * 3, 5, 3),
        )

        self.stg2 = nn.Sequential(
            XBlock(1024, 1024 * 3, 5, 3),
            YBlock(1024, 1024),
            XBlock(1024, 1024 * 3, 5, 3),
            YBlock(1024, 1024),
            nn.PixelShuffle(2),
        )

        self.stg3 = nn.Sequential(
            XBlock(512, 512 * 3, 5, 3),
            YBlock(512, 512),
            XBlock(512, 512 * 3, 5, 3),
            YBlock(512, 512),
            nn.PixelShuffle(2),
        )

        self.stg4 = nn.Sequential(
            XBlock(256, 256 * 3, 5, 3),
            YBlock(256, 256),
            XBlock(256, 256 * 3, 5, 3),
            YBlock(256, 256),
            nn.Conv2d(256, 16, 3, 1, 1),
        )

    def forward(self, x, d, c):
        c = torch.cat([self.dnet(d), c], 1)
        x = self.stg1(x)
        x = torch.cat([c, x], 1)
        x = self.stg2(x)
        x = torch.cat([self.con1(c), x], 1)
        x = self.stg3(x)
        x = torch.cat([self.con2(c), x], 1)
        x = self.stg4(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dnet = nn.PixelUnshuffle(16)

        self.fnet = nn.Sequential(
            XBlock(16, 16 * 3, 3, 1),
            nn.PixelUnshuffle(2),
            XBlock(64, 64 * 3, 3, 1),
            nn.PixelUnshuffle(2),
            XBlock(256, 256 * 3, 3, 1),
        )

        self.main = nn.Sequential(
            ZBlock(2304, 1024),

            XBlock(1024, 1024 * 3, 5, 3),
            ZBlock(1024, 512),
            
            XBlock(512, 512 * 3, 5, 3),
            ZBlock(512, 256),

            XBlock(256, 256 * 3, 5, 3),
            ZBlock(256, 128),

            nn.Conv2d(128, 1, 1, 1, 0)
        )

    def forward(self, cframe, cclip, lframe, lclip, tdepth, tclip):
        x = torch.cat([
            self.fnet(cframe),
            self.fnet(lframe),
            self.dnet(tdepth),
            cclip,
            lclip,
            tclip
        ], 1)
        return self.main(x)