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

from resnet import _ConvBatchNormReLU, _ResBlock, _ResBlockMG


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        self.stages.add_module(
            'c0',
            _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1),
        )
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i + 1),
                _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding, dilation),
            )
        self.imagepool = nn.Sequential(
            OrderedDict([
                ('pool', nn.AdaptiveAvgPool2d(1)),
                ('conv', _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)),
            ])
        )

    def forward(self, x):
        h = self.imagepool(x)
        h = [F.upsample(h, size=x.shape[2:], mode='bilinear')]
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h


class DeepLabV3(nn.Sequential):
    """DeepLab v3"""

    def __init__(self, n_classes, n_blocks, pyramids, multi_grid=[1, 2, 1]):
        super(DeepLabV3, self).__init__()
        self.add_module(
            'layer1',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                    ('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ])
            )
        )
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))  # output_stride=4
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))  # output_stride=8
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))  # output_stride=8
        self.add_module('layer5', _ResBlockMG(n_blocks[3], 1024, 512, 2048, 1, 2, mg=multi_grid))
        self.add_module('aspp', _ASPPModule(2048, 256, pyramids))
        self.add_module('fc1', _ConvBatchNormReLU(256 * (len(pyramids) + 2), 256, 1, 1, 0, 1))
        self.add_module('fc2', nn.Conv2d(256, n_classes, kernel_size=1))

    def forward(self, x):
        return super(DeepLabV3, self).forward(x)

    def freeze_bn(self):
        for m in self.named_modules():
            if 'layer' in m[0]:
                if isinstance(m[1], nn.BatchNorm2d):
                    print m[0]
                    m[1].eval()


if __name__ == '__main__':
    model = DeepLabV3(n_classes=21, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18])
    model.freeze_bn()
    model.eval()
    print list(model.named_children())
    image = torch.autograd.Variable(torch.randn(1, 3, 513, 513), volatile=True)
    print model(image)[0].size()
