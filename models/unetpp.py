import torch
import torch.nn as nn
from .blocks import DoubleConv, up_to


class UNetPlusPlus(nn.Module):
    """U-Net++ with an exposed bottleneck for SmokeLap fusion."""
    def __init__(self, in_channels=3, num_classes=9, base_channels=32):
        super().__init__()
        f = [base_channels, base_channels*2, base_channels*4, base_channels*8, base_channels*16]
        self.filters = f
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = DoubleConv(in_channels, f[0])
        self.conv1_0 = DoubleConv(f[0], f[1])
        self.conv2_0 = DoubleConv(f[1], f[2])
        self.conv3_0 = DoubleConv(f[2], f[3])
        self.conv4_0 = DoubleConv(f[3], f[4])

        self.conv0_1 = DoubleConv(f[0] + f[1], f[0])
        self.conv1_1 = DoubleConv(f[1] + f[2], f[1])
        self.conv2_1 = DoubleConv(f[2] + f[3], f[2])
        self.conv3_1 = DoubleConv(f[3] + f[4], f[3])

        self.conv0_2 = DoubleConv(f[0]*2 + f[1], f[0])
        self.conv1_2 = DoubleConv(f[1]*2 + f[2], f[1])
        self.conv2_2 = DoubleConv(f[2]*2 + f[3], f[2])

        self.conv0_3 = DoubleConv(f[0]*3 + f[1], f[0])
        self.conv1_3 = DoubleConv(f[1]*3 + f[2], f[1])

        self.conv0_4 = DoubleConv(f[0]*4 + f[1], f[0])
        self.final = nn.Conv2d(f[0], num_classes, kernel_size=1)

    def encode(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        return [x0_0, x1_0, x2_0, x3_0, x4_0]

    def decode(self, encoder_features, bottleneck=None):
        x0_0, x1_0, x2_0, x3_0, x4_original = encoder_features
        x4_0 = x4_original if bottleneck is None else bottleneck

        x0_1 = self.conv0_1(torch.cat([x0_0, up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, up_to(x3_0, x2_0)], dim=1))
        x3_1 = self.conv3_1(torch.cat([x3_0, up_to(x4_0, x3_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up_to(x2_1, x1_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, up_to(x3_1, x2_0)], dim=1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up_to(x1_2, x0_0)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, up_to(x2_2, x1_0)], dim=1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up_to(x1_3, x0_0)], dim=1))
        return self.final(x0_4)

    def forward(self, x):
        feats = self.encode(x)
        return self.decode(feats)
