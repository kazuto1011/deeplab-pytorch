#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-06

import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from libs.models import *

input = torch.randn(1, 3, 513, 513).cuda()

with SummaryWriter("runs/deeplab_v2", comment="DeepLabV2") as w:
    w.add_graph(DeepLabV2_ResNet101_MSC(n_classes=183).cuda(), (input,))

with SummaryWriter("runs/deeplab_v3", comment="DeepLabV3") as w:
    w.add_graph(DeepLabV3_ResNet101_MSC(n_classes=183).cuda(), (input,))

with SummaryWriter("runs/deeplab_v3_plus", comment="DeepLabV3+") as w:
    w.add_graph(DeepLabV3Plus_ResNet101_MSC(n_classes=183).cuda(), (input,))
