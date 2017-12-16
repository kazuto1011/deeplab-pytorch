#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

import glob
import os.path as osp
from collections import Counter, defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.misc as m
import torch
from torch.utils import data
from tqdm import tqdm
import torchvision
import random


class CocoStuff10k(data.Dataset):
    """
    COCO Stuff 10k
    """

    def __init__(self, root, split="train", image_size=512, scale=True, flip=True, preload=False):
        self.root = root
        self.split = split
        self.n_classes = 91 + 91 + 1
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)  # NOQA
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.files = defaultdict(list)
        self.scale = scale
        self.flip = flip
        self.preload = preload
        self.images = []
        self.labels = []
        self.ignore_label = 0

        # Load all path to images
        for split in ["train", "test", "all"]:
            file_list = tuple(
                open(root + '/imageLists/' + split + '.txt', 'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        if self.preload:
            self.preload_data()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        if self.preload:
            image, label = self.images[index], self.labels[index]
        else:
            image_id = self.files[self.split][index]
            image, label = self.load_pairwise(image_id)
        image, label = self.transform(image, label)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int32)

    def transform(self, image, label):
        if self.scale:
            scale_factor = random.uniform(0.5, 1.5)  # Hardware constraint
            image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, None, fx=scale_factor, fy=scale_factor,
                               interpolation=cv2.INTER_NEAREST)
            h, w = label.shape
            if scale_factor < 1.0:
                # Padding
                pad_h = max(self.image_size[0] - h, 0)
                pad_w = max(self.image_size[1] - w, 0)
                if pad_h > 0 or pad_w > 0:
                    image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(0.0, 0.0, 0.0))
                    label = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                               pad_w, cv2.BORDER_CONSTANT,
                                               value=(self.ignore_label,))
            else:
                # Random cropping
                off_h = random.randint(0, h - self.image_size[0])
                off_w = random.randint(0, w - self.image_size[1])
                image = image[off_h: off_h + self.image_size[0],
                              off_w: off_w + self.image_size[1]]
                label = label[off_h: off_h + self.image_size[0],
                              off_w: off_w + self.image_size[1]]
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()  # HWC
                label = np.flip(label, axis=1).copy()  # HW
        return image, label

    def load_pairwise(self, image_id):
        image_path = self.root + '/images/' + image_id + '.jpg'
        label_path = self.root + '/annotations/' + image_id + '.mat'
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, self.image_size,
                           interpolation=cv2.INTER_LINEAR)
        image -= self.mean
        # Load a label map
        label = sio.loadmat(label_path)['S'].astype(np.int32)
        label = cv2.resize(label, self.image_size,
                           interpolation=cv2.INTER_NEAREST)
        return image, label

    def preload_data(self):
        for image_id in tqdm(self.files[self.split],
                             desc='Preloading...',
                             leave=False,
                             dynamic_ncols=True):
            image, label = self.load_pairwise(image_id)
            self.images.append(image)
            self.labels.append(label)


if __name__ == '__main__':
    dataset_root = '/Users/kazuto1011/Desktop/cocostuff-10k-v1.1'
    batch_size = 64

    with open(osp.join(dataset_root, 'cocostuff-labels.txt')) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split(': ')
            classes[int(label[0])] = label[1]

    dataset = CocoStuff10k(root=dataset_root, split="test", preload=True)
    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, data in tqdm(enumerate(loader),
                        total=np.ceil(len(dataset) / batch_size),
                        leave=False):
        imgs, labels = data

        if i == 0:
            img = torchvision.utils.make_grid(imgs).numpy()
            img = np.transpose(img, (1, 2, 0)) + np.array([104.00699, 116.66877, 122.67892])
            img = img[:, :, ::-1].astype(np.uint8)
            for i, (word, cnt) in enumerate(Counter(labels.numpy().flatten()).most_common(10)):
                print i, classes[word]
            plt.imshow(img)
            plt.show()
            quit()
