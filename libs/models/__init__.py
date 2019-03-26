from __future__ import absolute_import
from .resnet import *
from .deeplabv1 import *
from .deeplabv2 import *
from .deeplabv3 import *
from .deeplabv3plus import *
from .msc import *


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def ResNet101(n_classes):
    return ResNet(n_classes=n_classes, n_blocks=[3, 4, 23, 3])


def DeepLabV1_ResNet101(n_classes):
    return DeepLabV1(n_classes=n_classes, n_blocks=[3, 4, 23, 3])


def DeepLabV2_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )


def DeepLabV2S_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[3, 6, 9, 12]
        ),
        scales=[0.5, 0.75],
    )


def DeepLabV3_ResNet101_MSC(n_classes, output_stride=16):
    if output_stride == 16:
        atrous_rates = [6, 12, 18]
    elif output_stride == 8:
        atrous_rates = [12, 24, 36]
    else:
        NotImplementedError

    base = DeepLabV3(
        n_classes=n_classes,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=atrous_rates,
        multi_grids=[1, 2, 4],
        output_stride=output_stride,
    )

    for name, module in base.named_modules():
        if ".bn" in name:
            module.momentum = 0.9997

    return MSC(base=base, scales=[0.5, 0.75])


def DeepLabV3Plus_ResNet101_MSC(n_classes, output_stride=16):
    if output_stride == 16:
        atrous_rates = [6, 12, 18]
    elif output_stride == 8:
        atrous_rates = [12, 24, 36]
    else:
        NotImplementedError

    base = DeepLabV3Plus(
        n_classes=n_classes,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=atrous_rates,
        multi_grids=[1, 2, 4],
        output_stride=output_stride,
    )

    for name, module in base.named_modules():
        if ".bn" in name:
            module.momentum = 0.9997

    return MSC(base=base, scales=[0.5, 0.75])
