#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   20 December 2018


def deeplabv2_resnet101(**kwargs):
    """
    DeepLab v2 model with ResNet-101 backbone
    n_classes (int): the number of classes
    """

    from libs.models.deeplabv2 import DeepLabV2
    from libs.models.msc import MSC

    base = DeepLabV2(n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24], **kwargs)
    model = MSC(scale=base, pyramids=[0.5, 0.75])

    return model
