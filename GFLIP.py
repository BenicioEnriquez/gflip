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
            XBlock(8, 8 * 4, 3, 3),

            nn.PixelUnshuffle(2),

            XBlock(32, 32 * 4, 3, 3),
            YBlock(32, 16),

            nn.PixelUnshuffle(2),

            XBlock(64, 64 * 4, 3, 3),

            nn.PixelUnshuffle(2),

            XBlock(256, 256 * 4, 3, 3),
            YBlock(256, 128),

            nn.PixelUnshuffle(2),

            XBlock(512, 512 * 4, 3, 3),

            nn.PixelUnshuffle(2),

            XBlock(2048, 2048 * 4, 3, 3),
            YBlock(2048, 1024),
        )

        self.mainblk = nn.Sequential(
            YBlock(3072, 4096),

            nn.PixelShuffle(2),

            YBlock(1024, 2048),
            
            nn.PixelShuffle(2),

            YBlock(512, 1024),
            YBlock(1024, 1024),
            
            nn.PixelShuffle(2),

            YBlock(256, 512),
            YBlock(512, 512),
            
            nn.PixelShuffle(2),

            YBlock(128, 256),
            YBlock(256, 256),
            YBlock(256, 256),

            nn.PixelShuffle(2),

            YBlock(64, 64),
            YBlock(64, 64),
            YBlock(64, 64),

            nn.Conv2d(64, 8, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, p, c1, c2, n):
        m = (p > 0.5).float()
        # x1 = torch.concat([x[:, :4] * m[:, :4] + n[:, :4] * (1 - m[:, :4]), p[:, :4]], dim=1)
        # x2 = torch.concat([x[:, -4:] * m[:, -4:] + n[:, -4:] * (1 - m[:, -4:]), p[:, -4:]], dim=1)
        x1 = torch.concat([x[:, :4] * m[:, :4] + n[:, :4] * (1 - m[:, :4]), (p[:, :4] * 2) - 1], dim=1)
        x2 = torch.concat([x[:, -4:] * m[:, -4:] + n[:, -4:] * (1 - m[:, -4:]), (p[:, -4:] * 2) - 1], dim=1)
        # x1 = torch.concat([x[:, :4] * m[:, :4], m[:, :4]], dim=1)
        # x2 = torch.concat([x[:, -4:] * m[:, -4:], m[:, -4:]], dim=1)
        x1 = self.headblk(x1)
        x2 = self.headblk(x2)
        x = torch.concat([x1, x2, c1, c2], dim=1)
        return self.mainblk(x)

