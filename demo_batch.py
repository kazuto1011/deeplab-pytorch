#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-01

import argparse
import os.path as osp
from glob import glob

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

    image_list = glob(osp.join(args.image_dir, '*'))

    for image_path in image_list:
        # Image preprocessing
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(float)
        image = cv2.resize(image, image_size)
        image_original = image.astype(np.uint8)[:, :, ::-1]
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

        rows = np.floor(np.sqrt(len(labels) + 1))
        cols = np.ceil((len(labels) + 1) / rows)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title('Input image')
        ax.imshow(image_original)
        ax.set_xticks([])
        ax.set_yticks([])

        for i, label in enumerate(labels):
            print '{0:3d}: {1}'.format(label, classes[label])
            mask = labelmap == label
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title(classes[label])
            # ax.imshow(np.dstack((mask,) * 3) * image_original)
            ax.imshow(mask)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.savefig('./data/results/{}'.format(osp.basename(image_path)))


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--dataset', nargs='?', type=str, default='cocostuff')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--image_dir', type=str, required=True)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    for arg in vars(args):
        print('{0:20s}: {1}'.format(arg.rjust(20), getattr(args, arg)))

    main(args)
