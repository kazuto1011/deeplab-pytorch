#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   19 February 2019

from __future__ import absolute_import, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import _ResLayer, _Stem


class DeepLabV1(nn.Sequential):
    """
    DeepLab v1: Dilated ResNet + 1x1 Conv
    Note that this is just a container for loading the pretrained COCO model and not mentioned as "v1" in papers.
    """

    def __init__(self, n_classes, n_blocks):
        super(DeepLabV1, self).__init__()
        ch = [64 * 2 ** p for p in range(6)]
        self.add_module("layer1", _Stem(ch[0]))
        self.add_module("layer2", _ResLayer(n_blocks[0], ch[0], ch[2], 1, 1))
        self.add_module("layer3", _ResLayer(n_blocks[1], ch[2], ch[3], 2, 1))
        self.add_module("layer4", _ResLayer(n_blocks[2], ch[3], ch[4], 1, 2))
        self.add_module("layer5", _ResLayer(n_blocks[3], ch[4], ch[5], 1, 4))
        self.add_module("fc", nn.Conv2d(2048, n_classes, 1))


if __name__ == "__main__":
    model = DeepLabV1(n_classes=21, n_blocks=[3, 4, 23, 3])
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).shape)
