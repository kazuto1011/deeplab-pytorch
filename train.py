#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.models import DeepLab
from libs.utils import CrossEntropyLoss2d


def get_1x_lr_params(model):
    b = []
    b.append(model.Scale.conv1)
    b.append(model.Scale.bn1)
    b.append(model.Scale.layer1)
    b.append(model.Scale.layer2)
    b.append(model.Scale.layer3)
    b.append(model.Scale.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            for k in j.parameters():
                jj += 1
                if k.requires_grad:
                    yield k


def get_10x_lr_params(model):
    b = []
    b.append(model.Scale.layer5.parameters())

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


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

    # Dataset
    dataset = get_dataset(args.dataset)(
        root=config['dataset'][args.dataset]['root'],
        split='train',
        image_size=(config['image']['size']['train'],
                    config['image']['size']['train']),
        scale=True,
        flip=True,
        # preload=True
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=config['num_workers'],
        shuffle=True
    )
    loader_iter = iter(loader)

    # Model
    model = DeepLab(n_classes=config['dataset'][args.dataset]['n_classes'])
    state_dict = torch.load(config['dataset'][args.dataset]['init_model'])
    if config['dataset'][args.dataset]['n_classes'] != 21:
        for i in state_dict:
            # 'Scale.layer5.conv2d_list.3.weight'
            i_parts = i.split('.')
            if i_parts[1] == 'layer5':
                state_dict[i] = model.state_dict()[i]
    model.load_state_dict(state_dict)
    if args.cuda:
        model.cuda()

    # Optimizer
    optimizer = {
        'sgd': torch.optim.SGD(
            params=[
                {'params': get_1x_lr_params(model), 'lr': args.lr},
                {'params': get_10x_lr_params(model), 'lr': 10 * args.lr}
            ],
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        ),
    }.get(args.optimizer)

    # Loss definition
    criterion = CrossEntropyLoss2d(
        ignore_index=config['dataset'][args.dataset]['ignore_label']
    )
    if args.cuda:
        criterion.cuda()

    # TensorBoard Logger
    writer = SummaryWriter(args.log_dir)
    loss_meter = MovingAverageValueMeter(20)

    model.train()
    optimizer.zero_grad()
    for iteration in tqdm(range(1, args.iter_max + 1),
                          total=args.iter_max,
                          leave=False,
                          dynamic_ncols=True):

        data, target = next(loader_iter)

        # Polynomial lr decay
        poly_lr_scheduler(optimizer=optimizer,
                          init_lr=args.lr,
                          iter=iteration - 1,
                          lr_decay_iter=args.lr_decay,
                          max_iter=args.iter_max,
                          power=args.poly_power)

        continue

        # Image
        data = data.cuda() if args.cuda else data
        data = Variable(data)

        # Forward propagation
        outputs = model(data)

        # Label
        target = resize_target(target, outputs[0].size(2))
        target = target.cuda() if args.cuda else target
        target = Variable(target)

        # Aggregate losses for [100%, 75%, 50%, Max]
        loss = 0
        for output in outputs:
            loss += criterion(output, target)
        loss /= args.iter_size
        loss.backward()
        loss_meter.add(loss.data[0])

        # Back propagation
        if iteration % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            loss = 0

        # TensorBoard
        if iteration % args.iter_tf == 0:
            writer.add_scalar('train_loss', loss_meter.value()[0], iteration)

        # Save a model
        if iteration % args.iter_snapshot == 0:
            torch.save(
                {'iteration': iteration,
                 'weight': model.state_dict()},
                osp.join(args.save_dir, 'checkpoint_{}.pth.tar'.format(iteration))
            )
            writer.add_text('log', 'Saved a model', iteration)

        if iteration % len(loader) == 0:
            loader_iter = iter(loader)

    torch.save(
        {'iteration': iteration,
         'weight': model.state_dict()},
        osp.join(args.save_dir, 'checkpoint_final.pth.tar')
    )


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, default='cocostuff')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--lr_decay', type=int, default=10)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--poly_power', type=float, default=0.9)
    parser.add_argument('--iter_max', type=int, default=20000)
    parser.add_argument('--iter_size', type=int, default=10)
    parser.add_argument('--iter_tf', type=int, default=50)
    parser.add_argument('--iter_snapshot', type=int, default=5000)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--log_dir', type=str, default='runs')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
