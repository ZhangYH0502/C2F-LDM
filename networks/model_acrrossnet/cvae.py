import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(CVAE, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        """encoder"""
        self.encoder_inconv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            DoubleConv(16, 32)
        )
        self.encoder_down1 = Down(32, 64)
        self.encoder_down2 = Down(64, 128)
        self.encoder_down3 = Down(128, 128)
        self.encoder_down4 = Down(128, 128)
        self.encoder_outconv = nn.Conv2d(128, 20, kernel_size=3, padding=1)
        """decoder"""
        self.decoder_inconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            DoubleConv(16, 32)
        )
        self.decoder_down1 = Down(32, 64)
        self.decoder_down2 = Down(64, 128)
        self.decoder_down3 = Down(128, 128)
        self.decoder_down4 = nn.MaxPool2d(2)
        self.up1 = Up(138, 128, 128)
        self.up2 = Up(256, 128, 128)
        self.up3 = Up(256, 128, 64)
        self.up4 = Up(128, 64, 32)
        self.decoder_outconv = nn.Sequential(
            DoubleConv(64, 32),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, xa1, ca2):
        """encoder"""
        enx0 = torch.cat([ca2, xa1], dim=1)
        enx1 = self.encoder_inconv(enx0)
        enx2 = self.encoder_down1(enx1)
        enx3 = self.encoder_down2(enx2)
        enx4 = self.encoder_down3(enx3)
        enx5 = self.encoder_down4(enx4)
        enx6 = self.encoder_outconv(enx5)

        mean = enx6[:, 0:10, :, :]  # mean
        logstd = enx6[:, 10:20, :, :]  # standard deviation
        za1 = self.reparametrize(mean, logstd)

        """decoder"""
        dex1 = self.decoder_inconv(ca2)
        dex2 = self.decoder_down1(dex1)
        dex3 = self.decoder_down2(dex2)
        dex4 = self.decoder_down3(dex3)
        dex5 = self.decoder_down4(dex4)
        dey0 = torch.cat([za1, dex5], dim=1)
        dey1 = self.up1(dey0, dex4)
        dey2 = self.up2(dey1, dex3)
        dey3 = self.up3(dey2, dex2)
        dey4 = self.up4(dey3, dex1)
        xxa1 = self.decoder_outconv(dey4)
        return mean, logstd, xxa1

    def reparametrize(self, mean, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)  # sample from standard normal distribution
        z = mean + eps * std
        return z


class DoubleConv(nn.Module):
    """(convolution => [IN] => LeakyReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
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
    """double conv then Upscaling"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.Sequential(
            DoubleConv(in_channels, out_channels1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels1, out_channels2, kernel_size=3, padding=1)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return x
