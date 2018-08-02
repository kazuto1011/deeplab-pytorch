#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2018-03-20

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict

from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("-m", "--model-path", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True, help="Use GPU/CPU.")
@click.option("--crf", is_flag=True, help="Apply CRF post processing.")
@click.option("--camera-id", type=int, default=0)
def main(config, model_path, cuda, crf, camera_id):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    # Label list
    with open(CONFIG.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]

    torch.set_grad_enabled(False)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    while True:
        # Image preprocessing
        ret, frame = cap.read()
        h, w, c = frame.shape
        image = frame.astype(np.float32)
        image -= np.array(
            [
                float(CONFIG.IMAGE.MEAN.B),
                float(CONFIG.IMAGE.MEAN.G),
                float(CONFIG.IMAGE.MEAN.R),
            ]
        )
        image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        image = image.to(device)

        # Inference
        output = model.scale(image)
        output = F.interpolate(output, size=(h, w), mode="bilinear")
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()[0]

        if crf:
            output = dense_crf(frame, output)
        labelmap = np.argmax(output.transpose(1, 2, 0), axis=2)

        labelmap = labelmap.astype(float) / CONFIG.N_CLASSES
        labelmap = cm.jet_r(labelmap)[..., :-1] * 255.0
        cv2.addWeighted(np.uint8(labelmap), 0.5, frame, 0.5, 0.0, frame)
        cv2.imshow("DeepLabV2", frame)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
