#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import argparse
import os.path as osp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.autograd import Variable

from libs.models import DeepLab
from libs.utils import dense_crf


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

    # Label list
    with open(config['dataset'][args.dataset]['label_list']) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split(': ')
            classes[int(label[0])] = label[1]

    # Path to a trained model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint,
                                map_location=lambda storage,
                                loc: storage)
        state_dict = checkpoint['weight']
        print('Result after {} iterations'.format(checkpoint['iteration']))
    else:
        state_dict = torch.load(
            config['dataset'][args.dataset]['trained_model'])

    # Model
    model = DeepLab(n_classes=config['dataset'][args.dataset]['n_classes'])
    model.load_state_dict(state_dict)
    model.eval()
    if args.cuda:
        model.cuda()

    image_size = (config['image']['size']['test'],
                  config['image']['size']['test'])

    # Image preprocessing
    image = cv2.imread(args.image, cv2.IMREAD_COLOR).astype(float)
    image = cv2.resize(image, image_size)
    image_original = image.astype(np.uint8)
    image -= np.array([config['image']['mean']['B'],
                       config['image']['mean']['G'],
                       config['image']['mean']['R']])
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.cuda() if args.cuda else image

    # Inference
    output = model(Variable(image, volatile=True))

    output = F.upsample(output[3], size=image_size, mode='bilinear')
    output = output[0].cpu().data.numpy()

    labelmap = dense_crf(image_original, output)
    labelmap = np.argmax(output.transpose(1, 2, 0), axis=2)

    labels = np.unique(labelmap)

    for i, label in enumerate(labels):
        print '{0:3d}: {1}'.format(label, classes[label])

    filteredmap = np.zeros(labelmap.shape)  # 0
    # Sky
    filteredmap[np.where(labelmap == 106)] = 100
    filteredmap[np.where(labelmap == 157)] = 100
    # Sea
    filteredmap[np.where(labelmap == 120)] = 200
    filteredmap[np.where(labelmap == 148)] = 200
    filteredmap[np.where(labelmap == 155)] = 200
    filteredmap[np.where(labelmap == 178)] = 200
    filteredmap[np.where(labelmap == 179)] = 200
    # Boat
    filteredmap[np.where(labelmap == 9)] = 255

    filteredmap = cv2.applyColorMap(np.uint8(filteredmap), cv2.COLORMAP_JET)
    filteredmap = filteredmap.astype(np.float) + image_original.astype(np.float)  # NOQA
    filteredmap = np.uint8(255 * filteredmap / filteredmap.max())

    plt.figure(figsize=(10, 10))

    ax = plt.subplot(1, 2, 1)
    ax.set_title('Input image')
    ax.imshow(image_original[:, :, ::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = plt.subplot(1, 2, 2)
    ax.set_title('Segmentation result')
    ax.imshow(filteredmap[:, :, ::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', type=str, default='cocostuff')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--image', type=str, required=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
