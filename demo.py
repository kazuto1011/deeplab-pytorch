#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.autograd import Variable

from models import DeepLab_ResNet
import os.path as osp


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

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
    model = DeepLab_ResNet(n_classes=config['dataset'][args.dataset]['n_classes'])
    model.load_state_dict(state_dict)
    model.eval()
    if args.cuda:
        model.cuda()

    image_size = (config['dataset'][args.dataset]['rows'],
                  config['dataset'][args.dataset]['cols'])

    # Image preprocessing
    image = cv2.imread(args.image, cv2.IMREAD_COLOR).astype(float)
    image = cv2.resize(image, image_size)
    image_original = image.astype(np.uint8)[:, :, ::-1]
    image -= np.array([config['mean']['B'],
                       config['mean']['G'],
                       config['mean']['R']])
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.cuda() if args.cuda else image

    # Inference
    output = model(Variable(image, volatile=True))

    output = F.upsample(output[3], size=image_size, mode='bilinear')
    output = output[0].cpu().data.numpy().transpose(1, 2, 0)
    labelmap = np.argmax(output, axis=2)

    # Class
    for label in np.unique(labelmap):
        print '{0:3d}: {1}'.format(label, classes[label])

    plt.subplot(1, 2, 1)
    plt.imshow(image_original)
    plt.subplot(1, 2, 2)
    plt.imshow(labelmap, cmap='gist_ncar')
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
