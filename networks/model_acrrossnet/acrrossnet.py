import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from networks.model_acrrossnet.unet import UNet
from networks.model_acrrossnet.cvae import CVAE


class ACRROSSNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super(ACRROSSNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.train_cvae = CVAE()
        self.train_CN = UNet()
        self.segmenter = nn.Sequential(
            DoubleConv(1, 64),
            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, xa1, xa2, xb):
        cb = self.train_CN(xb)
        pb = self.segmenter(cb)
        ca2 = self.train_CN(xa2)
        mean, logstd, xxa1 = self.train_cvae(xa1, ca2)

        return mean, logstd, xxa1, pb, cb


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


if __name__ == "__main__":
    """ set flags / seeds """
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    xa1 = torch.ones(8, 1, 304, 304).to(device)
    xa2 = torch.ones(8, 1, 304, 304).to(device)
    xb = torch.ones(2, 1, 304, 304).to(device)
    model = ACRROSSNet().to(device)
    mean, logstd, xxa1, pb, cb = model(xa1, xa2, xb)
    print(mean.shape)
    print(logstd.shape)
    print(xxa1.shape)
    print(pb.shape)
