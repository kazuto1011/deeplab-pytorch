#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import os.path as osp

import click
import cv2
import numpy as np
import yaml
from tqdm import tqdm

import torch
from libs.datasets import get_dataset
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import CrossEntropyLoss2d
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchnet.meter import MovingAverageValueMeter


def get_1x_lr_params(model):
    b = []
    b.append(model.scale.layer1)
    b.append(model.scale.layer2)
    b.append(model.scale.layer3)
    b.append(model.scale.layer4)
    b.append(model.scale.layer5)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.scale.aspp.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter, max_iter, power):
    if iter % lr_decay_iter or iter > max_iter:
        return None

    new_lr = init_lr * (1 - float(iter) / max_iter)**power
    optimizer.param_groups[0]['lr'] = new_lr
    optimizer.param_groups[1]['lr'] = 10 * new_lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.int32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)  # NOQA
    return torch.from_numpy(new_target).long()


@click.command()
@click.option('--config', type=str, default='config/cocostuff.yaml')
@click.option('--cuda/--no-cuda', default=True)
def main(config, cuda):
    # Configuration
    with open(config) as f:
        CONFIG = yaml.load(f)

    cuda = cuda and torch.cuda.is_available()

    # Dataset
    dataset = get_dataset(CONFIG['DATASET'])(
        root=CONFIG['ROOT'],
        split='train',
        image_size=(CONFIG['IMAGE']['SIZE']['TRAIN'],
                    CONFIG['IMAGE']['SIZE']['TRAIN']),
        scale=True,
        flip=True,
        # preload=True
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        shuffle=True
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG['N_CLASSES'])  # NOQA
    state_dict = torch.load(CONFIG['INIT_MODEL'])
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    if cuda:
        model.cuda()

    # Optimizer
    optimizer = {
        'sgd': torch.optim.SGD(
            params=[
                {'params': get_1x_lr_params(model), 'lr': float(CONFIG['LR'])},
                {'params': get_10x_lr_params(model), 'lr': 10 * float(CONFIG['LR'])}
            ],
            lr=float(CONFIG['LR']),
            momentum=float(CONFIG['MOMENTUM']),
            weight_decay=float(CONFIG['WEIGHT_DECAY'])
        ),
    }.get(CONFIG['OPTIMIZER'])

    # Loss definition
    criterion = CrossEntropyLoss2d(
        ignore_index=CONFIG['IGNORE_LABEL']
    )
    if cuda:
        criterion.cuda()

    # TensorBoard Logger
    writer = SummaryWriter(CONFIG['LOG_DIR'])
    loss_meter = MovingAverageValueMeter(20)

    model.train()
    for iteration in tqdm(range(1, CONFIG['ITER_MAX'] + 1),
                          total=CONFIG['ITER_MAX'],
                          leave=False,
                          dynamic_ncols=True):

        # Polynomial lr decay
        poly_lr_scheduler(optimizer=optimizer,
                          init_lr=float(CONFIG['LR']),
                          iter=iteration - 1,
                          lr_decay_iter=CONFIG['LR_DECAY'],
                          max_iter=CONFIG['ITER_MAX'],
                          power=CONFIG['POLY_POWER'])

        optimizer.zero_grad()

        iter_loss = 0
        for i in range(1, CONFIG['ITER_SIZE'] + 1):
            data, target = next(loader_iter)

            # Image
            data = data.cuda() if cuda else data
            data = Variable(data)

            # Forward propagation
            outputs = model(data)

            # Label
            target = resize_target(target, outputs[0].size(2))
            target = target.cuda() if cuda else target
            target = Variable(target)

            # Aggregate losses for [100%, 75%, 50%, Max]
            loss = 0
            for output in outputs:
                loss += criterion(output, target)

            loss /= CONFIG['ITER_SIZE']
            iter_loss += loss.data[0]
            loss.backward()

            # Reload dataloader
            if ((iteration - 1) * CONFIG['ITER_SIZE'] + i) % len(loader) == 0:
                loader_iter = iter(loader)

        loss_meter.add(iter_loss)

        # Back propagation
        optimizer.step()

        # TensorBoard
        if iteration % CONFIG['ITER_TF'] == 0:
            writer.add_scalar('train_loss', loss_meter.value()[0], iteration)

        # Save a model
        if iteration % CONFIG['ITER_SNAP'] == 0:
            torch.save(
                {'iteration': iteration,
                 'weight': model.state_dict()},
                osp.join(CONFIG['SAVE_DIR'],
                         'checkpoint_{}.pth.tar'.format(iteration))
            )
            writer.add_text('log', 'Saved a model', iteration)

    torch.save(
        {'iteration': iteration,
         'weight': model.state_dict()},
        osp.join(CONFIG['SAVE_DIR'], 'checkpoint_final.pth.tar')
    )


if __name__ == '__main__':
    main()
