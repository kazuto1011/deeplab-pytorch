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

from models import Res_Deeplab


def main(args):
    # Configuration
    with open(args.config) as f:
        config = yaml.load(f)

    # Path to a trained model
    if args.checkpoint:
        model_path = args.checkpoint
    else:
        model_path = config['dataset'][args.dataset]['trained_model']

    # Model
    model = Res_Deeplab(n_classes=config['dataset'][args.dataset]['n_classes'])
    model.load_state_dict(torch.load(model_path))
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
    for o in output:
        print o.size()

    output = F.upsample(output[3], size=image_size, mode='bilinear')
    output = output[0].cpu().data.numpy().transpose(1, 2, 0)
    labelmap = np.argmax(output, axis=2)

    # Class
    for label in np.unique(labelmap):
        print '{0:2d}: {1}'.format(label, config['dataset'][args.dataset]['classes'][label])

    plt.subplot(1, 2, 1)
    plt.imshow(image_original)
    plt.subplot(1, 2, 2)
    plt.imshow(labelmap)
    plt.show()


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', type=str, default='voc')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--image', type=str, required=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
