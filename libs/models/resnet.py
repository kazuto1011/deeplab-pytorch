#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            'bn',
            nn.BatchNorm2d(
                num_features=out_channels,
                eps=1e-5,
                momentum=0.999,
                affine=True,
            ),
        )

        if relu:
            self.add_module('relu', nn.ReLU())

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)


class _Bottleneck(nn.Sequential):
    """Bottleneck Unit"""

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBatchNormReLU(in_channels, mid_channels, 1, stride, 0, 1)
        self.conv3x3 = _ConvBatchNormReLU(mid_channels, mid_channels, 3, 1, dilation, dilation)
        self.increase = _ConvBatchNormReLU(mid_channels, out_channels, 1, 1, 0, 1, relu=False)
        self.downsample = downsample
        if self.downsample:
            self.proj = _ConvBatchNormReLU(in_channels, out_channels, 1, stride, 0, 1, relu=False)

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        if self.downsample:
            h += self.proj(x)
        else:
            h += x
        return F.relu(h)


class _ResBlock(nn.Sequential):
    """Residual Block"""

    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation):
        super(_ResBlock, self).__init__()
        self.add_module('block1', _Bottleneck(in_channels, mid_channels, out_channels, stride, dilation, True))
        for i in range(2, n_layers + 1):
            self.add_module('block' + str(i), _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation, False))

    def __call__(self, x):
        return super(_ResBlock, self).forward(x)


class _ResBlockMG(nn.Sequential):
    """3x Residual Block with multi-grid"""

    def __init__(self, n_layers, in_channels, mid_channels, out_channels, stride, dilation, mg=[1, 2, 1]):
        super(_ResBlockMG, self).__init__()
        self.add_module('block1', _Bottleneck(in_channels, mid_channels, out_channels, stride, dilation * mg[0], True))
        self.add_module('block2', _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation * mg[1], False))
        self.add_module('block3', _Bottleneck(out_channels, mid_channels, out_channels, 1, dilation * mg[2], False))

    def __call__(self, x):
        return super(_ResBlockMG, self).forward(x)
