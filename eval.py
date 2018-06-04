#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03

from __future__ import absolute_import, division, print_function

import json
import os.path as osp

import click
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import CocoStuff10k
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf, scores


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("-m", "--model-path", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--crf", is_flag=True)
def main(config, model_path, cuda, crf):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    image_size = (CONFIG.IMAGE.SIZE.TEST,) * 2

    # Dataset
    dataset = CocoStuff10k(
        root=CONFIG.ROOT,
        split="test",
        image_size=image_size,
        scale=False,
        flip=False,
        preload=False,
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    torch.set_grad_enabled(False)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    model.load_state_dict(
        torch.load(model_path, map_location=lambda storage, loc: storage)
    )
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    targets, outputs = [], []
    for data, target in tqdm(
        loader, total=len(loader), leave=False, dynamic_ncols=True
    ):
        # Image
        data = data.to(device)

        # Forward propagation
        output = model(data)
        output = F.upsample(
            output, size=image_size, mode="bilinear", align_corners=False
        )
        output = F.softmax(output, dim=1)
        output = output.data.cpu().numpy()

        # Postprocessing
        if crf:
            crf_output = np.zeros(output.shape)
            images = data.data.cpu().numpy().astype(np.uint8)
            for i, (image, prob_map) in enumerate(zip(images, output)):
                image = image.transpose(1, 2, 0)
                crf_output[i] = dense_crf(image, prob_map)
            output = crf_output

        output = np.argmax(output, axis=1)
        target = target.numpy()

        for o, t in zip(output, target):
            outputs.append(o)
            targets.append(t)

    score, class_iou = scores(targets, outputs, n_class=CONFIG.N_CLASSES)

    for k, v in score.items():
        print(k, v)

    score["Class IoU"] = {}
    for i in range(CONFIG.N_CLASSES):
        score["Class IoU"][i] = class_iou[i]

    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
