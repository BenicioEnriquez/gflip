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

        self.headblk = nn.PixelUnshuffle(4)

        self.mainblk = nn.Sequential(

            XBlock(2304, 2304 * 3, 7, 5),
            YBlock(2304, 3072),
            XBlock(3072, 3072 * 3, 7, 5),
            YBlock(3072, 4096),

            nn.PixelShuffle(2),

            XBlock(1024, 1024 * 3, 5, 3),
            YBlock(1024, 1536),
            XBlock(1536, 1536 * 3, 5, 3),
            YBlock(1536, 2048),
            
            nn.PixelShuffle(2),

            XBlock(512, 512 * 3, 5, 3),
            YBlock(512, 512),

            nn.Conv2d(512, 40, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x, m, c1, c2):
        x = torch.concat([x * m, m], dim=1)
        x = self.headblk(x)
        x = torch.concat([x, c1, c2], dim=1)
        return self.mainblk(x)

