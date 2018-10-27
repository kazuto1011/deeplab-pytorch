#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03

from __future__ import absolute_import, division, print_function

import json
import multiprocessing as mp

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf, scores


def dense_crf_wrapper(args):
    return dense_crf(args[0], args[1])


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("-m", "--model-path", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
@click.option("--crf", is_flag=True)
def main(config, model_path, cuda, crf):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    CONFIG = Dict(yaml.load(open(config)))

    # Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET)(
        root=CONFIG.ROOT,
        split=CONFIG.SPLIT.VAL,
        base_size=CONFIG.IMAGE.SIZE.TEST,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.WARP_IMAGE,
        scale=None,
        flip=False,
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.BATCH_SIZE.TEST,
        num_workers=CONFIG.NUM_WORKERS,
        shuffle=False,
    )

    torch.set_grad_enabled(False)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    preds, gts = [], []
    for images, labels in tqdm(
        loader, total=len(loader), leave=False, dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)
        logits = F.interpolate(
            logits, size=images.shape[2:], mode="bilinear", align_corners=True
        )
        probs = F.softmax(logits, dim=1)
        probs = probs.data.cpu().numpy()

        # Postprocessing
        if crf:
            pool = mp.Pool(mp.cpu_count())
            images = images.data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            probs = pool.map(dense_crf_wrapper, zip(images, probs))
            pool.close()

        preds += list(np.argmax(probs, axis=1))
        gts += list(labels.numpy())

    score = scores(gts, preds, n_class=CONFIG.N_CLASSES)

    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
