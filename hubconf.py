#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   20 December 2018

from __future__ import print_function

from torch.hub import load_state_dict_from_url

model_url_root = "https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/"
model_dict = {
    "cocostuff10k": ("deeplabv2_resnet101_msc-cocostuff10k-20000.pth", 182),
    "cocostuff164k": ("deeplabv2_resnet101_msc-cocostuff164k-100000.pth", 182),
    "voc12": ("deeplabv2_resnet101_msc-vocaug-20000.pth", 21),
}


def deeplabv2_resnet101(pretrained=None, n_classes=182, scales=None):

    from libs.models.deeplabv2 import DeepLabV2
    from libs.models.msc import MSC

    # Model parameters
    n_blocks = [3, 4, 23, 3]
    atrous_rates = [6, 12, 18, 24]
    if scales is None:
        scales = [0.5, 0.75]

    base = DeepLabV2(n_classes=n_classes, n_blocks=n_blocks, atrous_rates=atrous_rates)
    model = MSC(base=base, scales=scales)

    # Load pretrained models
    if isinstance(pretrained, str):

        assert pretrained in model_dict, list(model_dict.keys())
        expected = model_dict[pretrained][1]
        error_message = "Expected: n_classes={}".format(expected)
        assert n_classes == expected, error_message

        model_url = model_url_root + model_dict[pretrained][0]
        state_dict = load_state_dict_from_url(model_url)
        model.load_state_dict(state_dict)

    return model

