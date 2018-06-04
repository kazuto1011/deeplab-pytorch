#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-26

import torch
import torch.nn as nn
import torch.nn.functional as F


class MSC(nn.Module):
    """Multi-scale inputs"""

    def __init__(self, model, pyramids=[0.5, 0.75]):
        super(MSC, self).__init__()
        self.scale = model
        self.pyramids = pyramids

    def forward(self, x):
        # Original
        logits = self.scale(x)
        interp = nn.Upsample(
            size=logits.shape[2:], mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.pyramids:
            size = [int(s * p) for s in x.shape[2:]]
            h = F.upsample(x, size=size, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.scale(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        if self.training:
            return [logits] + logits_pyramid + [logits_max]
        else:
            return logits_max
