#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019


from __future__ import absolute_import, division, print_function

import json
import os.path as osp

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from addict import Dict
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR, scores


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def setup_model(model_path, n_classes, train=True):
    model = DeepLabV2_ResNet101_MSC(n_classes=n_classes)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    if train:
        model.load_state_dict(state_dict, strict=False)  # to skip ASPP
        model = nn.DataParallel(model)
    else:
        model.load_state_dict(state_dict)
        model = nn.DataParallel(model)
        model.eval()
    return model


def get_params(model, key):
    # For Dilated FCN
    if key == "1x":
        for m in model.named_modules():
            if "layer" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
    # For conv weight in the ASPP module
    if key == "10x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].weight
    # For conv bias in the ASPP module
    if key == "20x":
        for m in model.named_modules():
            if "aspp" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    yield m[1].bias


def resize_labels(labels, shape):
    labels = labels.unsqueeze(1).float()  # Add channel axis
    labels = F.interpolate(labels, shape, mode="nearest")
    labels = labels.squeeze(1).long()
    return labels


@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option("-c", "--config", type=str, required=True, help="yaml")
@click.option("--cuda/--no-cuda", default=True, help="Switch GPU/CPU")
def train(config, cuda):
    # Auto-tune cuDNN
    torch.backends.cudnn.benchmark = True

    # Configuration
    device = get_device(cuda)
    CONFIG = Dict(yaml.load(open(config)))

    # Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        base_size=CONFIG.IMAGE.SIZE.TRAIN.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN.CROP,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.DATASET.WARP_IMAGE,
        scale=CONFIG.DATASET.SCALES,
        flip=True,
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model
    model = setup_model(CONFIG.MODEL.INIT_MODEL, CONFIG.DATASET.N_CLASSES, train=True)
    model.to(device)

    # Optimizer
    optimizer = torch.optim.SGD(
        # cf lr_mult and decay_mult in train.prototxt
        params=[
            {
                "params": get_params(model.module, key="1x"),
                "lr": CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="10x"),
                "lr": 10 * CONFIG.SOLVER.LR,
                "weight_decay": CONFIG.SOLVER.WEIGHT_DECAY,
            },
            {
                "params": get_params(model.module, key="20x"),
                "lr": 20 * CONFIG.SOLVER.LR,
                "weight_decay": 0.0,
            },
        ],
        momentum=CONFIG.SOLVER.MOMENTUM,
    )

    # Learning rate scheduler
    scheduler = PolynomialLR(
        optimizer=optimizer,
        step_size=CONFIG.SOLVER.LR_DECAY,
        iter_max=CONFIG.SOLVER.ITER_MAX,
        power=CONFIG.SOLVER.POLY_POWER,
    )

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

    # TensorBoard logger
    writer = SummaryWriter(CONFIG.SOLVER.LOG_DIR)
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        leave=False,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                images, labels = next(loader_iter)

            images = images.to(device)
            labels = labels.to(device)

            # Propagate forward
            logits = model(images)

            # Loss
            iter_loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, shape=(H, W))
                iter_loss += criterion(logit, labels_)

            # Backpropagate (just compute gradients wrt the loss)
            iter_loss /= CONFIG.SOLVER.ITER_SIZE
            iter_loss.backward()

            loss += float(iter_loss)

        average_loss.add(loss)

        # Update weights with accumulated gradients
        optimizer.step()

        # Update learning rate
        scheduler.step(epoch=iteration)

        # TensorBoard
        if iteration % CONFIG.SOLVER.ITER_TB == 0:
            writer.add_scalar("loss/train", average_loss.value()[0], iteration)
            for i, o in enumerate(optimizer.param_groups):
                writer.add_scalar("lr/group{}".format(i), o["lr"], iteration)
            if False:  # This produces a large log file
                for name, param in model.named_parameters():
                    name = name.replace(".", "/")
                    # Weight/gradient distribution
                    writer.add_histogram(name, param, iteration, bins="auto")
                    if param.requires_grad:
                        writer.add_histogram(
                            name + "/grad", param.grad, iteration, bins="auto"
                        )

        # Save a model
        if iteration % CONFIG.SOLVER.ITER_SAVE == 0:
            torch.save(
                model.module.state_dict(),
                osp.join(CONFIG.MODEL.SAVE_DIR, "checkpoint_{}.pth".format(iteration)),
            )

        # To verify progress separately
        torch.save(
            model.module.state_dict(),
            osp.join(CONFIG.MODEL.SAVE_DIR, "checkpoint_current.pth"),
        )

    torch.save(
        model.module.state_dict(),
        osp.join(CONFIG.MODEL.SAVE_DIR, "checkpoint_final.pth"),
    )


@main.command()
@click.option("-c", "--config", type=str, required=True, help="yaml")
@click.option("-m", "--model-path", type=str, required=True, help="pth")
@click.option("--cuda/--no-cuda", default=True, help="Switch GPU/CPU")
@click.option("--crf", is_flag=True, help="CRF post processing")
def test(config, model_path, cuda, crf):
    # Disable autograd globally
    torch.set_grad_enabled(False)

    # Setup
    device = get_device(cuda)
    CONFIG = Dict(yaml.load(open(config)))

    # If the image size never change,
    if CONFIG.DATASET.WARP_IMAGE:
        # Auto-tune cuDNN
        torch.backends.cudnn.benchmark = True

    # Dataset 10k or 164k
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        base_size=CONFIG.IMAGE.SIZE.TEST,
        crop_size=None,
        mean=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        warp=CONFIG.DATASET.WARP_IMAGE,
        scale=None,
        flip=False,
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = setup_model(model_path, CONFIG.DATASET.N_CLASSES, train=False)
    model.to(device)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    preds, gts = [], []
    for images, labels in tqdm(
        loader, total=len(loader), leave=False, dynamic_ncols=True
    ):
        # Image
        images = images.to(device)
        _, H, W = labels.shape

        # Forward propagation
        logits = model(images)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
        probs = F.softmax(logits, dim=1)
        probs = probs.data.cpu().numpy()

        # Postprocessing
        if crf:
            # images: (B,C,H,W) -> (B,H,W,C)
            images = images.data.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
            probs = joblib.Parallel(n_jobs=-1)(
                [joblib.delayed(postprocessor)(*pair) for pair in zip(images, probs)]
            )

        labelmaps = np.argmax(probs, axis=1)

        preds += list(labelmaps)
        gts += list(labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(model_path.replace(".pth", ".json"), "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
