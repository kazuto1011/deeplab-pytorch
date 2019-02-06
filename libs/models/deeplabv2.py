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

from .resnet import _ConvBnReLU, _ResLayer, _Stem


class _ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(zip(rates)):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class DeepLabV2(nn.Sequential):
    """DeepLab v2 (OS=8)"""

    def __init__(self, n_classes, n_blocks, atrous_rates):
        super(DeepLabV2, self).__init__()
        self.add_module("layer1", _Stem())
        self.add_module("layer2", _ResLayer(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], 1024, 512, 2048, 1, 4))
        self.add_module("aspp", _ASPP(2048, n_classes, atrous_rates))

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, _ConvBnReLU.BATCH_NORM):
                m.eval()


if __name__ == "__main__":
    model = DeepLabV2(
        n_classes=21, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
