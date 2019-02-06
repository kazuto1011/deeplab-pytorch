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

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d


class _ConvBnReLU(nn.Sequential):
    BATCH_NORM = _BATCH_NORM

    def __init__(
        self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True
    ):
        super(_ConvBnReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False
            ),
        )
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))

        if relu:
            self.add_module("relu", nn.ReLU())


class _Bottleneck(nn.Module):
    """Bottleneck Unit"""

    def __init__(self, in_ch, mid_ch, out_ch, stride, dilation, downsample):
        super(_Bottleneck, self).__init__()
        self.reduce = _ConvBnReLU(in_ch, mid_ch, 1, stride, 0, 1, True)
        self.conv3x3 = _ConvBnReLU(mid_ch, mid_ch, 3, 1, dilation, dilation, True)
        self.increase = _ConvBnReLU(mid_ch, out_ch, 1, 1, 0, 1, False)
        self.shortcut = (
            _ConvBnReLU(in_ch, out_ch, 1, stride, 0, 1, False)
            if downsample
            else lambda x: x  # identity
        )

    def forward(self, x):
        h = self.reduce(x)
        h = self.conv3x3(h)
        h = self.increase(h)
        h += self.shortcut(x)
        return F.relu(h)


class _ResLayer(nn.Sequential):
    """Residual blocks"""

    def __init__(
        self, n_layers, in_ch, mid_ch, out_ch, stride, dilation, multi_grids=None
    ):
        super(_ResLayer, self).__init__()

        if multi_grids is None:
            multi_grids = [1 for _ in range(n_layers)]
        else:
            assert n_layers == len(
                multi_grids
            ), "{} values expected, but got: mg={}".format(n_layers, multi_grids)

        self.add_module(
            "block1",
            _Bottleneck(in_ch, mid_ch, out_ch, stride, dilation * multi_grids[0], True),
        )
        for i, rate in zip(range(2, n_layers + 1), multi_grids[1:]):
            self.add_module(
                "block" + str(i),
                _Bottleneck(out_ch, mid_ch, out_ch, 1, dilation * rate, False),
            )


class _Stem(nn.Sequential):
    """
    The 1st Residual Layer
    """

    def __init__(self):
        super(_Stem, self).__init__()
        self.add_module("conv1", _ConvBnReLU(3, 64, 7, 2, 3, 1))
        self.add_module("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True))
