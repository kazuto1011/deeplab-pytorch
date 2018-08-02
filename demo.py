#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
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
@click.option("-i", "--image-path", type=str, required=True)
@click.option("-m", "--model-path", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--crf", is_flag=True, help="Apply CRF post processing.")
def main(config, image_path, model_path, cuda, crf):
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

    # Image preprocessing
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    image_original = image.astype(np.uint8)
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
    output = model(image)
    output = F.interpolate(output, size=image.shape[2:], mode="bilinear")
    output = F.softmax(output, dim=1)
    output = output.data.cpu().numpy()[0]

    if crf:
        output = dense_crf(image_original, output)
    labelmap = np.argmax(output, axis=0)

    labels = np.unique(labelmap)

    # Show results
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Input image")
    ax.imshow(image_original[:, :, ::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    for i, label in enumerate(labels):
        print("{0:3d}: {1}".format(label, classes[label]))
        mask = labelmap == label
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(image_original[..., ::-1])
        ax.imshow(mask.astype(np.float32), alpha=0.5, cmap="viridis")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
