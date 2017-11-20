#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15


import click
import cv2
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.models import DeepLabV2_ResNet101_MSC
from libs.utils import dense_crf
from torch.autograd import Variable


@click.command()
@click.option('--dataset', required=True, type=click.Choice(['voc12']))
@click.option('--image_path', required=True)
@click.option('--model_path', default=None)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--crf', is_flag=True)
def main(dataset, image_path, model_path, cuda, crf):
    CONFIG = {
        'voc12': {
            'path_pytorch_model': 'data/models/deeplab_resnet101/voc12/deeplabv2_resnet101_VOC2012.pth',
            'label_list': 'data/datasets/voc12/labels.txt',
            'n_classes': 21,
            'n_blocks': [3, 4, 23, 3],
            'pyramids': [6, 3, 2, 1],
            'image': {
                'size': {
                    'train': 473,
                    'test': 473,
                },
                'mean': {
                    'R': 122.675,
                    'G': 116.669,
                    'B': 104.008,
                }
            },
        },
    }.get(dataset)

    cuda = cuda and torch.cuda.is_available()

    # Label list
    with open(CONFIG['label_list']) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split('\t')
            classes[int(label[0])] = label[1].split(',')[0]

    # Load a model
    if model_path is None:
        state_dict = torch.load(CONFIG['path_pytorch_model'])
    else:
        state_dict = torch.load(model_path)

    # Model
    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG['n_classes'])
    model.load_state_dict(state_dict)
    model.eval()
    if cuda:
        model.cuda()

    image_size = (CONFIG['image']['size']['test'],
                  CONFIG['image']['size']['test'])

    # Image preprocessing
    image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
    image = cv2.resize(image, image_size)
    image_original = image.astype(np.uint8)
    image -= np.array([float(CONFIG['image']['mean']['B']),
                       float(CONFIG['image']['mean']['G']),
                       float(CONFIG['image']['mean']['R'])])
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.cuda() if cuda else image

    # Inference
    output = model(Variable(image, volatile=True))

    output = F.upsample(output, size=image_size, mode='bilinear')
    output = F.softmax(output)
    output = output.data.cpu().numpy()[0]

    if crf:
        output = dense_crf(image_original, output)
    labelmap = np.argmax(output.transpose(1, 2, 0), axis=2)

    labels = np.unique(labelmap)

    # Show results
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title('Input image')
    ax.imshow(image_original[:, :, ::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    for i, label in enumerate(labels):
        print '{0:3d}: {1}'.format(label, classes[label])
        mask = labelmap == label
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(np.dstack((mask,) * 3) * image_original[:, :, ::-1])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
    main()
