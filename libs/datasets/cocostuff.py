#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-10-30

from __future__ import print_function

import glob
import os.path as osp
import random
from collections import defaultdict

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data
from tqdm import tqdm

_MEAN = [104.008, 116.669, 122.675]

_VERSION = "1.1"


class CocoStuff10k(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        image_size=513,
        crop_size=321,
        scale=True,
        flip=True,
        preload=False,
    ):
        self.root = root
        self.split = split
        self.image_size = (
            image_size if isinstance(image_size, tuple) else (image_size, image_size)
        )
        self.crop_size = (
            crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)
        )
        self.scale = scale  # scale and crop
        self.flip = flip
        self.preload = preload
        self.mean = np.array(_MEAN)
        self.files = defaultdict(list)
        self.images = []
        self.labels = []
        self.ignore_label = -1

        # Load all path to images
        for split in ["train", "test", "all"]:
            file_list = tuple(open(root + "/imageLists/" + split + ".txt", "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files[split] = file_list

        if self.preload:
            self._preload_data()

        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        if self.preload:
            image, label = self.images[index], self.labels[index]
        else:
            image_id = self.files[self.split][index]
            image, label = self._load_data(image_id)
        image, label = self._transform(image, label)
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int64)

    def _transform(self, image, label):
        if self.scale:
            # Scaling
            scale_factor = random.uniform(0.5, 1.5)
            image = cv2.resize(
                image,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            label = cv2.resize(
                label,
                None,
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv2.INTER_NEAREST,
            )
            h, w = label.shape
            # Padding
            if scale_factor < 1.0:
                pad_h = max(self.image_size[0] - h, 0)
                pad_w = max(self.image_size[1] - w, 0)
                if pad_h > 0 or pad_w > 0:
                    image = cv2.copyMakeBorder(
                        image,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        cv2.BORDER_CONSTANT,
                        value=(0.0, 0.0, 0.0),
                    )
                    label = cv2.copyMakeBorder(
                        label,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        cv2.BORDER_CONSTANT,
                        value=(self.ignore_label,),
                    )
            # Random cropping
            h, w = label.shape
            off_h = random.randint(0, h - self.crop_size[0])
            off_w = random.randint(0, w - self.crop_size[1])
            image = image[
                off_h : off_h + self.crop_size[0], off_w : off_w + self.crop_size[1]
            ]
            label = label[
                off_h : off_h + self.crop_size[0], off_w : off_w + self.crop_size[1]
            ]
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def _load_data(self, image_id):
        image_path = self.root + "/images/" + image_id + ".jpg"
        label_path = self.root + "/annotations/" + image_id + ".mat"
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image -= self.mean
        # Load a label map
        if _VERSION == "1.1":
            label = sio.loadmat(label_path)["S"].astype(np.int64)
            label -= 1  # unlabeled (0 -> -1)
        else:
            label = np.array(h5py.File(label_path, "r")["S"], dtype=np.int64).transpose(
                1, 0
            )
            label -= 2  # unlabeled (1 -> -1)
        label = cv2.resize(label, self.image_size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def _preload_data(self):
        for image_id in tqdm(
            self.files[self.split],
            desc="Preloading...",
            leave=False,
            dynamic_ncols=True,
        ):
            image, label = self._load_data(image_id)
            self.images.append(image)
            self.labels.append(label)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Version: {}\n".format(_VERSION)
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    from torchvision.utils import make_grid

    dataset_root = "/media/kazuto1011/Extra/cocostuff/cocostuff-10k-v" + _VERSION
    kwargs = {"nrow": 10, "padding": 30}
    batch_size = 100

    dataset = CocoStuff10k(root=dataset_root, split="train")
    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, data in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):
        imgs, labels = data

        if i == 0:
            mean = torch.tensor(_MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            imgs += mean.expand_as(imgs)
            img = make_grid(imgs, **kwargs).numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img[:, :, ::-1].astype(np.uint8)

            label = make_grid(
                labels[:, np.newaxis, ...], pad_value=-1, **kwargs
            ).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32) + 1
            label = cm.jet(label_ / 183.)[..., :3] * 255
            label *= (label_ != 0)[..., None]
            label = label.astype(np.uint8)

            img = np.hstack((img, label))
            plt.figure(figsize=(20, 20))
            plt.imshow(img)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig("./docs/data.png", bbox_inches="tight", transparent=True)
            plt.show()
            quit()
