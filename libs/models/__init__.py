from __future__ import absolute_import
from .resnet import *
from .deeplabv2 import *
from .deeplabv3 import *
from .deeplabv3plus import *
from .msc import *


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


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


def DeepLabV3_ResNet101_MSC(n_classes, output_stride):
    if output_stride == 16:
        atrous_rates = [6, 12, 18]
    elif output_stride == 8:
        atrous_rates = [12, 24, 36]
    else:
        NotImplementedError

    return MSC(
        base=DeepLabV3(
            n_classes=n_classes,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=atrous_rates,
            multi_grids=[1, 2, 4],
            output_stride=output_stride,
        ),
        scales=[0.5, 0.75],
    )


def DeepLabV3Plus_ResNet101_MSC(n_classes, output_stride):
    if output_stride == 16:
        atrous_rates = [6, 12, 18]
    elif output_stride == 8:
        atrous_rates = [12, 24, 36]
    else:
        NotImplementedError

    return MSC(
        base=DeepLabV3Plus(
            n_classes=n_classes,
            n_blocks=[3, 4, 23, 3],
            atrous_rates=atrous_rates,
            multi_grids=[1, 2, 4],
            output_stride=output_stride,
        ),
        scales=[0.5, 0.75],
    )
