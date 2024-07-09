""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
from networks.mynet_parts.normlizations import get_norm_layer
from networks.mynet_parts.activations import get_activations

class UNet(nn.Module):
    def __init__(self, n_in=1, n_out=1, first_channels=32, n_dps=4, use_bilinear=True, use_pool=True, active_type='relu',
                 norm_type='instance3D'):
        super(UNet, self).__init__()
        activation = get_activations(active_type)
        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder(n_in, first_channels, n_dps, use_pool, norm_layer, activation)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder(n_out, first_channels, n_dps, use_bilinear, norm_layer, activation)

    def forward(self, x):
        features = self.encoder(x)
        # for feature in features:
        #     print(feature.shape)
        out = self.decoder(features)
        return out
class UNet_2out(nn.Module):
    def __init__(self, n_in=1, n_out=1, first_channels=32, n_dps=4, use_bilinear=True, use_pool=True, active_type='relu',
                 norm_type='instance3D', **kwargs):
        super(UNet_2out, self).__init__()
        activation = get_activations(active_type)
        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder(n_in, first_channels, n_dps, use_pool, norm_layer, activation)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder_2out(n_out, first_channels, n_dps, use_bilinear, norm_layer, activation)

    def forward(self, x):
        features = self.encoder(x)
        # for feature in features:
        #     print(feature.shape)
        out = self.decoder(features)
        return out

class UNet_3out(nn.Module):
    def __init__(self, n_in=1, n_out=1, first_channels=32, n_dps=4, use_bilinear=True, use_pool=True, active_type='relu',
                 norm_type='instance3D', **kwargs):
        super(UNet_3out, self).__init__()
        activation = get_activations(active_type)
        norm_layer = get_norm_layer(norm_type)

        self.encoder = UnetEncoder(n_in, first_channels, n_dps, use_pool, norm_layer, activation)
        first_channels = first_channels * pow(2, n_dps)
        self.decoder = UnetDecoder_3out(n_out, first_channels, n_dps, use_bilinear, norm_layer, activation)

    def forward(self, x):
        features = self.encoder(x)
        # for feature in features:
        #     print(feature.shape)
        out = self.decoder(features)
        return out

class UnetEncoder(nn.Module):
    def __init__(self, in_channels, first_channels, n_dps, use_pool, norm_layer, activation):
        super(UnetEncoder, self).__init__()
        self.inc = InConv(in_channels, first_channels, norm_layer, activation)
        self.down_blocks = nn.ModuleList()
        in_channels = first_channels
        out_channels = in_channels * 2
        for i in range(n_dps):
            self.down_blocks.append(Down(in_channels, out_channels, use_pool, norm_layer, activation))
            in_channels = out_channels
            out_channels = in_channels * 2

    def forward(self, x):
        x = self.inc(x)
        out_features = [x]
        for down_block in self.down_blocks:
            x = down_block(x)
            out_features.append(x)
        return out_features


class UnetDecoder(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer, activation):
        super(UnetDecoder, self).__init__()

        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer, activation))
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)

        self.is_out_features = False

    def forward(self, features):
        pos_feat = len(features) - 1
        x = features[pos_feat]
        out_features = [x]
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
            out_features.append(x)
        x = self.outc(x)
        if self.is_out_features:
            return x, out_features
        else:
            return x
class UnetDecoder_2out(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer, activation):
        super(UnetDecoder_2out, self).__init__()

        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer, activation))
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc1 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.outc2 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.is_out_features = False

    def forward(self, features):
        pos_feat = len(features) - 1
        x = features[pos_feat]
        out_features = [x]
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
            out_features.append(x)
        x1 = self.outc1(x)
        x2 = self.outc2(x)
        if self.is_out_features:
            return x1,x2, out_features
        else:
            return x1,x2

class UnetDecoder_3out(nn.Module):
    def __init__(self, n_classes, first_channels, n_dps, use_bilinear, norm_layer, activation):
        super(UnetDecoder_3out, self).__init__()

        self.up_blocks = nn.ModuleList()
        T_channels = first_channels
        out_channels = T_channels // 2
        in_channels = T_channels + out_channels

        for i in range(n_dps):
            self.up_blocks.append(Up(T_channels, in_channels, out_channels, use_bilinear, norm_layer, activation))
            T_channels = out_channels
            out_channels = T_channels // 2
            in_channels = T_channels + out_channels
        # one more divide in out_channels
        self.outc1 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.outc2 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.outc3 = nn.Conv3d(out_channels*2, n_classes, kernel_size=1)
        self.is_out_features = False

    def forward(self, features):
        pos_feat = len(features) - 1
        x = features[pos_feat]
        out_features = [x]
        for up_block in self.up_blocks:
            pos_feat -= 1
            x = up_block(x, features[pos_feat])
            out_features.append(x)
        x1 = self.outc1(x)
        x2 = self.outc2(x)
        x3 = self.outc3(x)
        if self.is_out_features:
            return x1,x2, x3, out_features
        else:
            return x1,x2, x3


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, activation):
        super().__init__()
        self.double_conv = nn.Sequential(
            norm_layer(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)),
            activation,
            norm_layer(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)),
            activation
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool, norm_layer, activation):
        super().__init__()
        if use_pool:
            self.down_conv = nn.Sequential(
                nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
                norm_layer(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)),
                activation,
                norm_layer(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)),
                activation
            )
        else:
            self.down_conv = nn.Sequential(
                norm_layer(nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                activation,
                norm_layer(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)),
                activation
            )

    def forward(self, x):
        return self.down_conv(x)


class Up(nn.Module):
    def __init__(self, T_channels, in_channels, out_channels, bilinear, norm_layer, activation):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=(1,2,2), mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(T_channels, T_channels, kernel_size=4, stride=2)
        self.conv = nn.Sequential(
            norm_layer(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)),
            activation,
            norm_layer(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)),
            activation
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # diffZ = x2.size()[2] - x1.size()[2]
        # diffY = x2.size()[3] - x1.size()[3]
        # diffX = x2.size()[4] - x1.size()[4]
        # # print(x1.shape, x2.shape)
        # if diffX> 0 or diffY> 0 or diffZ > 0:
        #     x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ//2, diffZ-diffZ//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
