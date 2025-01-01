""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from einops import rearrange
from torch import Tensor, LongTensor
from timm.models.layers import DropPath, Mlp
from torchsummary import summary
from torch.nn import Conv2d
from einops.layers.torch import Rearrange, Reduce
# from tensorboardX import SummaryWriters
import numpy as np
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(2 * in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CNN_Down(nn.Module):
    def __init__(self, dim, downsample=None):
        super().__init__()
        self.dim = dim
        if downsample is not None:
            self.blocks = Down(in_channels=dim, out_channels=2 * dim)
        else:
            self.blocks = None

    def forward(self, x):
        if self.blocks is not None:
            x = self.blocks(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# class LocalAttention(nn.Module):
#
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.C = in_channels
#         self.O = out_channels
#         assert in_channels == out_channels
#         self.dconv5_5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
#         self.dconv1_3 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), padding=(0, 1), groups=in_channels)
#         self.dconv3_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), padding=(1, 0), groups=in_channels)
#         self.dconv1_5 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), padding=(0, 2), groups=in_channels)
#         self.dconv5_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), padding=(2, 0), groups=in_channels)
#         self.dconv1_7 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 7), padding=(0, 3), groups=in_channels)
#         self.dconv7_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(7, 1), padding=(3, 0), groups=in_channels)
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, inputs):
#         #   Global Perceptron
#         inputs = self.conv(inputs)
#         inputs = self.act(inputs)
#
#         channel_att_vec = self.ca(inputs)
#         inputs = channel_att_vec * inputs
#
#         x_init = self.dconv5_5(inputs)
#         x_1 = self.dconv1_3(x_init)
#         x_1 = self.dconv3_1(x_1)
#         x_2 = self.dconv1_5(x_init)
#         x_2 = self.dconv5_1(x_2)
#         x_3 = self.dconv1_7(x_init)
#         x_3 = self.dconv7_1(x_3)
#         x = x_1 + x_2 + x_3 + x_init
#         spatial_att = self.conv(x)
#         out = spatial_att * inputs
#         out = self.sigmoid(out)
#         out = self.conv(out)
#         return out
class MSFF(nn.Module):
    def __init__(self, ch_1, ch_2, r_2, ch_int, ch_out, drop_rate=0.):
        super(HFF_block, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // r_2, 1,bias=False),
            nn.ReLU(),
            nn.Conv2d(ch_2 // r_2, ch_2, 1,bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_l = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_g = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.Avg = nn.AvgPool2d(2, stride=2)
        self.Updim = Conv(ch_int//2, ch_int, 1, bn=True, relu=True)
        self.norm1 = LayerNorm(ch_int * 3, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(ch_int * 2, eps=1e-6, data_format="channels_first")
        self.norm3 = LayerNorm(ch_1 + ch_2 + ch_int, eps=1e-6, data_format="channels_first")
        self.W3 = Conv(ch_int * 3, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int * 2, ch_int, 1, bn=True, relu=False)
        self.dconv5_5 = nn.Conv2d(ch_int, ch_int, kernel_size=5, padding=2, groups=ch_int)
        self.dconv1_3 = nn.Conv2d(ch_int, ch_int, kernel_size=(1, 3), padding=(0, 1), groups=ch_int)
        self.dconv3_1 = nn.Conv2d(ch_int, ch_int, kernel_size=(3, 1), padding=(1, 0), groups=ch_int)
        self.dconv1_5 = nn.Conv2d(ch_int, ch_int, kernel_size=(1, 5), padding=(0, 2), groups=ch_int)
        self.dconv5_1 = nn.Conv2d(ch_int, ch_int, kernel_size=(5, 1), padding=(2, 0), groups=ch_int)
        self.dconv1_7 = nn.Conv2d(ch_int, ch_int, kernel_size=(1, 7), padding=(0, 3), groups=ch_int)
        self.dconv7_1 = nn.Conv2d(ch_int, ch_int, kernel_size=(7, 1), padding=(3, 0), groups=ch_int)
        self.conv = nn.Conv2d(ch_int, ch_out, kernel_size=(1, 1), padding=0)
        self.gelu = nn.GELU()

        self.residual = IRMLP(ch_1 + ch_2 + ch_int, ch_out)
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, l, g, f):
        g_1 = torch.transpose(g, 1, 2)
        B, C, new_HW = g_1.shape
        g_1 = g_1.view(g_1.shape[0], C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        W_local = self.W_l(l)   # local feature from Local Feature Block
        W_global = self.W_g(g_1)   # global feature from Global Feature Block
        if f is not None:
            W_f = self.Updim(f)
            W_f = self.Avg(W_f)
            shortcut = W_f
            X_f = torch.cat([W_f, W_local, W_global], 1)
            X_f = self.norm1(X_f)
            X_f = self.W3(X_f)
            X_f = self.gelu(X_f)
        else:
            shortcut = 0
            X_f = torch.cat([W_local, W_global], 1)
            X_f = self.norm2(X_f)
            X_f = self.W(X_f)
            X_f = self.gelu(X_f)

        # spatial attention for ConvNeXt branch
        l_init = self.dconv5_5(l)
        l_1 = self.dconv1_3(l_init)
        l_1 = self.dconv3_1(l_1)
        l_2 = self.dconv1_5(l_init)
        l_2 = self.dconv5_1(l_2)
        l_3 = self.dconv1_7(l_init)
        l_3 = self.dconv7_1(l_3)
        l = l_1 + l_2 + l_3 + l_init
        l = self.sigmoid(l) * l_init
        l = self.conv(l)

        # channel attetion for transformer branch
        g_jump = g_1
        max_result=self.maxpool(g_1)
        avg_result=self.avgpool(g_1)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        g = self.sigmoid(max_out+avg_out) * g_jump

        fuse = torch.cat([g, l, X_f], 1)
        fuse = self.norm3(fuse)
        fuse = self.residual(fuse)
        fuse = shortcut + self.drop_path(fuse)
        return fuse

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#### Inverted Residual MLP
class IRMLP(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(IRMLP, self).__init__()
        self.conv1 = Conv(inp_dim, inp_dim, 3, relu=False, bias=False, group=inp_dim)
        self.conv2 = Conv(inp_dim, inp_dim * 4, 1, relu=False, bias=False)
        self.conv3 = Conv(inp_dim * 4, out_dim, 1, relu=False, bias=False, bn=True)
        self.gelu = nn.GELU()
        self.bn1 = nn.BatchNorm2d(inp_dim)

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.gelu(out)
        out += residual

        out = self.bn1(out)
        out = self.conv2(out)
        out = self.gelu(out)
        out = self.conv3(out)

        return out


