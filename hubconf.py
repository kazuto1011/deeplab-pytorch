#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   20 December 2018

from __future__ import print_function


def deeplabv2_resnet101(pretrained=False, **kwargs):
    """
    DeepLab v2 model with ResNet-101 backbone
    n_classes (int): the number of classes
    """

    if pretrained:
        raise NotImplementedError(
            "Please download from "
            "https://github.com/kazuto1011/deeplab-pytorch/tree/master#performance"
        )

    from libs.models.deeplabv2 import DeepLabV2
    from libs.models.msc import MSC

    base = DeepLabV2(n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24], **kwargs)
    model = MSC(base=base, scales=[0.5, 0.75])

    return model


if __name__ == "__main__":
    import torch.hub

    model = torch.hub.load(
        "kazuto1011/deeplab-pytorch",
        "deeplabv2_resnet101",
        n_classes=182,
        force_reload=True,
    )

    print(model)
