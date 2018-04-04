#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn


class MSC(nn.Module):
    """Multi-Scale Inputs"""

    def __init__(self, model):
        super(MSC, self).__init__()
        self.scale = model

    def forward(self, x):
        output100 = self.scale(x)
        input_size = x.size(2)
        size100 = output100.size(2)
        size075 = int(input_size * 0.75)
        size050 = int(input_size * 0.5)

        self.interp075 = nn.Upsample(size=(size075, ) * 2, mode='bilinear')
        self.interp050 = nn.Upsample(size=(size050, ) * 2, mode='bilinear')
        self.interp100 = nn.Upsample(size=(size100, ) * 2, mode='bilinear')

        output075 = self.scale(self.interp075(x))
        output050 = self.scale(self.interp050(x))

        outputMAX = torch.max(
            torch.stack((
                output100,
                self.interp100(output075),
                self.interp100(output050),
            )), dim=0
        )[0]

        if self.training:
            return [output100, output075, output050, outputMAX]
        else:
            return outputMAX