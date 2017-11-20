#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03


import argparse
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchnet.meter import MovingAverageValueMeter
from tqdm import tqdm

from libs.datasets import get_dataset
from libs.utils import dense_crf
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import CrossEntropyLoss2d, scores


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

    image_size = (config[args.dataset]['image']['size']['test'],
                  config[args.dataset]['image']['size']['test'])
    n_classes = config[args.dataset]['n_classes']

    # Dataset
    dataset = get_dataset(args.dataset)(
        root=config[args.dataset]['root'],
        split='test',
        image_size=image_size,
        scale=False,
        flip=False,
        preload=False
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=config['num_workers'],
        shuffle=False
    )

    checkpoint = torch.load(args.checkpoint,
                            map_location=lambda storage,
                            loc: storage)
    state_dict = checkpoint['weight']
    print('Result after {} iterations'.format(checkpoint['iteration']))

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=n_classes)
    # model.load_state_dict(state_dict)
    model.eval()
    if args.cuda:
        model.cuda()

    targets, outputs = [], []
    for data, target in tqdm(loader, total=len(loader),
                             leave=False, dynamic_ncols=True):
        # Image
        data = data.cuda() if args.cuda else data
        data = Variable(data, volatile=True)

        # Forward propagation
        output = model(data)
        output = F.upsample(output, size=image_size, mode='bilinear')
        output = F.softmax(output)
        output = output.data.cpu().numpy()

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

    score, class_iou = scores(targets, outputs, n_class=n_classes)

    for k, v in score.items():
        print k, v

    for i in range(n_classes):
        print i, class_iou[i]


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', type=str, default='cocostuff')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
