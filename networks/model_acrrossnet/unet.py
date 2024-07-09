""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, ret_mid=False):
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)
        x6 = self.up1(x5, x4)
        # print(x6.shape)
        x7 = self.up2(x6, x3)
        # print(x7.shape)
        x8 = self.up3(x7, x2)
        # print(x8.shape)
        x9 = self.up4(x8, x1)
        # print(x9.shape)
        logits = self.outc(x9)
        if ret_mid:
            return logits, [x1, x2, x3, x4, x5, x6, x7, x8, x9]
        return logits



class merge_net(nn.Module):
    def __init__(self, n_classes=1, bilinear=False):
        super(merge_net, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(129, 128)
        self.down1 = Down_merge(384, 256)
        self.down2 = Down_merge(768, 512)
        self.down3 = Down_merge(1536, 1024)
        self.down4 = Down_merge(2048, 2048)
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.outc = OutConv(128, n_classes)

    def forward(self, x, mids):
        y1,y2,y3,y4,y5,y6,y7,y8,y9=mids
        f1 = torch.cat([x, y1, y9], dim=1)
        f2 = torch.cat([y2, y8], dim=1)
        f3 = torch.cat([y3, y7], dim=1)
        f4 = torch.cat([y4, y6], dim=1)
        f5 = y5
        x1 = self.inc(f1)
        x2 = self.down1(x1, f2)
        x3 = self.down2(x2, f3)
        x4 = self.down3(x3, f4)
        # print(x4.shape, f5.shape)
        x5 = self.down4(x4, f5)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        logits = self.outc(x9)
        return logits


class Down_merge(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, y):
        x = self.max_pool(x)
        return self.conv(torch.cat([x,y], dim=1))

class DoubleConv(nn.Module):
    """(convolution => [IN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
