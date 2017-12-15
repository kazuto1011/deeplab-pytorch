#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-03


import os.path as osp

import click
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
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import CrossEntropyLoss2d, dense_crf, scores


@click.command()
@click.option('--config', type=str, default='config/cocostuff.yaml')
@click.option('--model-path', type=str, required=True)
@click.option('--cuda/--no-cuda', default=True)
def main(config, model_path, cuda):
    # Configuration
    with open(config) as f:
        CONFIG = yaml.load(f)

    cuda = cuda and torch.cuda.is_available()

    image_size = (CONFIG['IMAGE']['SIZE']['TEST'],
                  CONFIG['IMAGE']['SIZE']['TEST'])
    n_classes = CONFIG['N_CLASSES']

    # Dataset
    dataset = get_dataset(CONFIG['DATASET'])(
        root=CONFIG['ROOT'],
        split='test',
        image_size=image_size,
        scale=False,
        flip=False,
        preload=False
    )

    # DataLoader
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        shuffle=False
    )

    state_dict = torch.load(model_path,
                            map_location=lambda storage,
                            loc: storage)
    state_dict = state_dict['weight']
    # print('Result after {} iterations'.format(state_dict['iteration']))

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=n_classes)
    # model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()

    targets, outputs = [], []
    for data, target in tqdm(loader, total=len(loader),
                             leave=False, dynamic_ncols=True):
        # Image
        data = data.cuda() if cuda else data
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
    main()
