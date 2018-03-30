from libs.models.resnet import *
from libs.models.deeplabv2 import *
from libs.models.deeplabv3 import *
from libs.models.deeplabv3plus import *
from libs.models.msc import *


def DeepLabV2_ResNet101_MSC(n_classes):
    return MSC(DeepLabV2(n_classes=n_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24]))


def DeepLabV2S_ResNet101_MSC(n_classes):
    return MSC(DeepLabV2(n_classes=n_classes, n_blocks=[3, 4, 23, 3], pyramids=[3, 6, 9, 12]))


def DeepLabV3_ResNet101_MSC(n_classes):
    return MSC(DeepLabV3(n_classes=n_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18]))


def DeepLabV3Plus_ResNet101_MSC(n_classes):
    return MSC(DeepLabV3Plus(n_classes=n_classes, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18]))