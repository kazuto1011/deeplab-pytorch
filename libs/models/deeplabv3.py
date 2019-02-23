#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling with image-level feature
    """

    def __init__(self, n_in, n_out, rates):
        super(_ASPP, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module("c0", _ConvBnReLU(n_in, n_out, 1, 1, 0, 1))
        for i, rate in enumerate(rates):
            self.stages.add_module(
                "c{}".format(i + 1),
                _ConvBnReLU(n_in, n_out, 3, 1, padding=rate, dilation=rate),
            )
        self.imagepool = nn.Sequential(
            OrderedDict(
                [
                    ("pool", nn.AdaptiveAvgPool2d(1)),
                    ("conv", _ConvBnReLU(n_in, n_out, 1, 1, 0, 1)),
                ]
            )
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Sequential):
    """
    DeepLab v3: Dilated ResNet with multi-grid + improved ASPP
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3, self).__init__()

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        self.add_module("layer1", _Stem())
        self.add_module(
            "layer2", _ResLayer(n_blocks[0], 64, 64, 256, stride[0], dilation[0])
        )
        self.add_module(
            "layer3", _ResLayer(n_blocks[1], 256, 128, 512, stride[1], dilation[1])
        )
        self.add_module(
            "layer4", _ResLayer(n_blocks[2], 512, 256, 1024, stride[2], dilation[2])
        )
        self.add_module(
            "layer5",
            _ResLayer(
                n_blocks[3], 1024, 512, 2048, stride[3], dilation[3], multi_grids
            ),
        )
        self.add_module("aspp", _ASPP(2048, 256, atrous_rates))
        self.add_module(
            "fc1", _ConvBnReLU(256 * (len(atrous_rates) + 2), 256, 1, 1, 0, 1)
        )
        self.add_module("fc2", nn.Conv2d(256, n_classes, kernel_size=1))


if __name__ == "__main__":
    model = DeepLabV3(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=8,
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
