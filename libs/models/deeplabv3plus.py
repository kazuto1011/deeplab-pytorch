#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeplabv3 import _ASPPModule
from .resnet import _ConvBatchNormReLU, _ResBlock


class DeepLabV3Plus(nn.Module):
    """DeepLab v3+ (OS=8)"""

    def __init__(self, n_classes, n_blocks, pyramids, grids, output_stride):
        super(DeepLabV3Plus, self).__init__()

        if output_stride == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 2, 2]
        elif output_stride == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]

        # Encoder
        self.add_module(
            "layer1",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                        ("pool", nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                    ]
                )
            ),
        )
        self.add_module(
            "layer2", _ResBlock(n_blocks[0], 64, 64, 256, stride[0], dilation[0])
        )
        self.add_module(
            "layer3", _ResBlock(n_blocks[1], 256, 128, 512, stride[1], dilation[1])
        )
        self.add_module(
            "layer4", _ResBlock(n_blocks[2], 512, 256, 1024, stride[2], dilation[2])
        )
        self.add_module(
            "layer5",
            _ResBlock(n_blocks[3], 1024, 512, 2048, stride[3], dilation[3], mg=grids),
        )
        self.add_module("aspp", _ASPPModule(2048, 256, pyramids))
        self.add_module(
            "fc1", _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1)
        )
        # Decoder
        self.add_module("reduce", _ConvBatchNormReLU(256, 48, 1, 1, 0, 1))
        self.add_module(
            "fc2",
            nn.Sequential(
                OrderedDict(
                    [
                        ("conv1", _ConvBatchNormReLU(304, 256, 3, 1, 1, 1)),
                        ("conv2", _ConvBatchNormReLU(256, 256, 3, 1, 1, 1)),
                        ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                    ]
                )
            ),
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h_ = self.reduce(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear")
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear")
        return h

    def freeze_bn(self):
        for m in self.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()


if __name__ == "__main__":
    model = DeepLabV3Plus(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        pyramids=[6, 12, 18],
        grids=[1, 2, 4],
        output_stride=16,
    )
    model.freeze_bn()
    model.eval()
    print(list(model.named_children()))
    image = torch.randn(1, 3, 513, 513)
    print(model(image)[0].size())
