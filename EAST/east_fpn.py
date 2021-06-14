
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class DeConvBNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None,
                 name=None):
        super(DeConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class EASTFPN(nn.Module):
    def __init__(self, in_channels, model_name, **kwargs):
        super(EASTFPN, self).__init__()
        self.model_name = model_name
        if self.model_name == "large":
            self.out_channels = 128
        else:
            self.out_channels = 64
        self.in_channels = in_channels[::-1]
        self.h1_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[1],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_1")
        self.h2_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[2],
            out_channels=self.out_channels,
            kernel_size=4,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_2")
        self.h3_conv = ConvBNLayer(
            in_channels=self.out_channels+self.in_channels[3],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_h_3")
        self.g0_deconv = DeConvBNLayer(
            in_channels=self.in_channels[0],
            out_channels=self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_0")
        self.g1_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=(3,4),
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_1")
        self.g2_deconv = DeConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=6,
            stride=2,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_2")
        self.g3_conv = ConvBNLayer(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            if_act=True,
            act='relu',
            name="unet_g_3")

    def forward(self, x):
        f = x[::-1]
        
        h = f[0]
        print(h.shape)
        g = self.g0_deconv(h)
        print(g.shape , f[1].shape)
        h = torch.cat([g, f[1]], axis=1)
        print(h.shape)
        h = self.h1_conv(h)
        print(h.shape)
        g = self.g1_deconv(h)
        print(g.shape, f[2].shape)
        h = torch.cat([g, f[2]], axis=1)
        print(h.shape)
        h = self.h2_conv(h)
        print(h.shape)
        g = self.g2_deconv(h)
        print(g.shape, f[3].shape)
        h = torch.cat([g, f[3]], axis=1)
        print(h.shape)
        h = self.h3_conv(h)
        print(h.shape)
        g = self.g3_conv(h)

        return g