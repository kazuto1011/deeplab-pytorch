#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os

import click
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import DenseCRF, PolynomialLR, scores


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        print("Device:")
        for i in range(torch.cuda.device_count()):
            print("    {}:".format(i), torch.cuda.get_device_name(i))
    else:
        print("Device: CPU")
    return device


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


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


@click.group()
@click.pass_context
def main(ctx):
    """
    Training and evaluation
    """
    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def train(config_path, cuda):
    """
    Training DeepLab by v2 protocol
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.backends.cudnn.benchmark = True

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.TRAIN,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=True,
        base_size=CONFIG.IMAGE.SIZE.BASE,
        crop_size=CONFIG.IMAGE.SIZE.TRAIN,
        scales=CONFIG.DATASET.SCALES,
        flip=True,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TRAIN,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=True,
    )
    loader_iter = iter(loader)

    # Model check
    print("Model:", CONFIG.MODEL.NAME)
    assert (
        CONFIG.MODEL.NAME == "DeepLabV2_ResNet101_MSC"
    ), 'Currently support only "DeepLabV2_ResNet101_MSC"'

    # Model setup
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(CONFIG.MODEL.INIT_MODEL)
    print("    Init:", CONFIG.MODEL.INIT_MODEL)
    for m in model.base.state_dict().keys():
        if m not in state_dict.keys():
            print("    Skip init:", m)
    model.base.load_state_dict(state_dict, strict=False)  # to skip ASPP
    model = nn.DataParallel(model)
    model.to(device)

    # Loss definition
    criterion = nn.CrossEntropyLoss(ignore_index=CONFIG.DATASET.IGNORE_LABEL)
    criterion.to(device)

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

    # Setup loss logger
    writer = SummaryWriter(os.path.join(CONFIG.EXP.OUTPUT_DIR, "logs", CONFIG.EXP.ID))
    average_loss = MovingAverageValueMeter(CONFIG.SOLVER.AVERAGE_LOSS)

    # Path to save models
    checkpoint_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "models",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.TRAIN,
    )
    makedirs(checkpoint_dir)
    print("Checkpoint dst:", checkpoint_dir)

    # Freeze the batch norm pre-trained on COCO
    model.train()
    model.module.base.freeze_bn()

    for iteration in tqdm(
        range(1, CONFIG.SOLVER.ITER_MAX + 1),
        total=CONFIG.SOLVER.ITER_MAX,
        dynamic_ncols=True,
    ):

        # Clear gradients (ready to accumulate)
        optimizer.zero_grad()

        loss = 0
        for _ in range(CONFIG.SOLVER.ITER_SIZE):
            try:
                _, images, labels = next(loader_iter)
            except:
                loader_iter = iter(loader)
                _, images, labels = next(loader_iter)

            # Propagate forward
            logits = model(images.to(device))

            # Loss
            iter_loss = 0
            for logit in logits:
                # Resize labels for {100%, 75%, 50%, Max} logits
                _, _, H, W = logit.shape
                labels_ = resize_labels(labels, size=(H, W))
                iter_loss += criterion(logit, labels_.to(device))

            # Propagate backward (just compute gradients)
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
                writer.add_scalar("lr/group_{}".format(i), o["lr"], iteration)
            for i in range(torch.cuda.device_count()):
                writer.add_scalar(
                    "gpu/device_{}/memory_cached".format(i),
                    torch.cuda.memory_cached(i) / 1024 ** 3,
                    iteration,
                )

            if False:
                for name, param in model.module.base.named_parameters():
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
                os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(iteration)),
            )

    torch.save(
        model.module.state_dict(), os.path.join(checkpoint_dir, "checkpoint_final.pth")
    )


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
def test(config_path, model_path, cuda):
    """
    Evaluation on validation set
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG.SOLVER.BATCH_SIZE.TEST,
        num_workers=CONFIG.DATALOADER.NUM_WORKERS,
        shuffle=False,
    )

    # Model
    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model = nn.DataParallel(model)
    model.eval()
    model.to(device)

    # Path to save logits
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    makedirs(logit_dir)
    print("Logit dst:", logit_dir)

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores.json")
    print("Score dst:", save_path)

    preds, gts = [], []
    for image_ids, images, gt_labels in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        # Image
        images = images.to(device)

        # Forward propagation
        logits = model(images)

        # Save on disk for CRF post-processing
        for image_id, logit in zip(image_ids, logits):
            filename = os.path.join(logit_dir, image_id + ".npy")
            np.save(filename, logit.cpu().numpy())

        # Pixel-wise labeling
        _, H, W = gt_labels.shape
        logits = F.interpolate(
            logits, size=(H, W), mode="bilinear", align_corners=False
        )
        probs = F.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)

        preds += list(labels.cpu().numpy())
        gts += list(gt_labels.numpy())

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-j",
    "--n-jobs",
    type=int,
    default=multiprocessing.cpu_count(),
    show_default=True,
    help="Number of parallel jobs",
)
def crf(config_path, n_jobs):
    """
    CRF post-processing on pre-computed logits
    """

    # Configuration
    CONFIG = OmegaConf.load(config_path)
    torch.set_grad_enabled(False)
    print("# jobs:", n_jobs)

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )
    print(dataset)

    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )

    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )
    print("Logit src:", logit_dir)
    if not os.path.isdir(logit_dir):
        print("Logit not found, run first: python main.py test [OPTIONS]")
        quit()

    # Path to save scores
    save_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "scores",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
    )
    makedirs(save_dir)
    save_path = os.path.join(save_dir, "scores_crf.json")
    print("Score dst:", save_path)

    # Process per sample
    def process(i):
        image_id, image, gt_label = dataset.__getitem__(i)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8).transpose(1, 2, 0)
        prob = postprocessor(image, prob)
        label = np.argmax(prob, axis=0)

        return label, gt_label

    # CRF in multi-process
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
        [joblib.delayed(process)(i) for i in range(len(dataset))]
    )

    preds, gts = zip(*results)

    # Pixel Accuracy, Mean Accuracy, Class IoU, Mean IoU, Freq Weighted IoU
    score = scores(gts, preds, n_class=CONFIG.DATASET.N_CLASSES)

    with open(save_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
