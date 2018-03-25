#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-20

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from torch.autograd import Variable

from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf


@click.command()
@click.option('--config', '-c', type=str, required=True)
@click.option('--model-path', '-m', type=str, required=True)
@click.option('--cuda/--no-cuda', default=True, help='Use GPU/CPU.')
@click.option('--crf', is_flag=True, help='Apply CRF post processing.')
@click.option('--camera-id', type=int, default=0)
def main(config, model_path, cuda, crf, camera_id):
    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    cuda = cuda and torch.cuda.is_available()
    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on', torch.cuda.get_device_name(current_device))

    # Label list
    with open(CONFIG.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split('\t')
            classes[int(label[0])] = label[1].split(',')[0]

    # Load a model
    state_dict = torch.load(model_path)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()

    image_size = (CONFIG.IMAGE.SIZE.TEST, ) * 2

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))

    while True:
        # Image preprocessing
        ret, frame = cap.read()
        image = cv2.resize(frame.astype(float), image_size)
        raw_image = image.astype(np.uint8)
        image -= np.array([
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ])
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.cuda() if cuda else image

        # Inference
        output = model(Variable(image, volatile=True))
        output = F.upsample(output, size=image_size, mode='bilinear')
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()[0]

        if crf:
            output = dense_crf(raw_image, output)
        labelmap = np.argmax(output.transpose(1, 2, 0), axis=2)

        labelmap = labelmap.astype(float) / CONFIG.N_CLASSES
        labelmap = cm.jet_r(labelmap)[..., :-1] * 255.0
        cv2.addWeighted(np.uint8(labelmap), 0.5, raw_image, 0.5, 0.0, raw_image)
        cv2.imshow('DeepLabV2', raw_image)
        cv2.waitKey(50)


if __name__ == '__main__':
    main()
