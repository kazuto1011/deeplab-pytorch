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


class CocoStuff10k(data.Dataset):
    """COCO-Stuff 10k dataset"""

    def __init__(
        self,
        root,
        split="train",
        base_size=513,
        crop_size=321,
        mean=(104.008, 116.669, 122.675),
        scale=(0.5, 1.5),
        warp=True,
        flip=True,
        preload=False,
        version="1.1",
    ):
        self.root = root
        self.split = split
        self.base_size = base_size
        self.crop_size = crop_size
        self.mean = np.array(mean)
        self.scale = scale
        self.warp = warp
        self.flip = flip
        self.preload = preload
        self.version = version

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
        return image.astype(np.float32), label.astype(np.int64)

    def _transform(self, image, label):
        # Mean subtraction
        image -= self.mean
        # Pre-scaling
        if self.warp:
            base_size = (self.base_size,) * 2
        else:
            raw_h, raw_w = label.shape
            if raw_h > raw_w:
                base_size = (int(self.base_size * raw_w / raw_h), self.base_size)
            else:
                base_size = (self.base_size, int(self.base_size * raw_h / raw_w))
        image = cv2.resize(image, base_size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, base_size, interpolation=cv2.INTER_NEAREST)
        if self.scale is not None:
            # Scaling
            scale_factor = random.uniform(self.scale[0], self.scale[1])
            scale_kwargs = {"dsize": None, "fx": scale_factor, "fy": scale_factor}
            image = cv2.resize(image, interpolation=cv2.INTER_LINEAR, **scale_kwargs)
            label = cv2.resize(label, interpolation=cv2.INTER_NEAREST, **scale_kwargs)
            scale_h, scale_w = label.shape
            # Padding
            pad_h = max(max(base_size[1], self.crop_size) - scale_h, 0)
            pad_w = max(max(base_size[0], self.crop_size) - scale_w, 0)
            pad_kwargs = {
                "top": pad_h // 2,
                "bottom": pad_h - pad_h // 2,
                "left": pad_w // 2,
                "right": pad_w - pad_w // 2,
                "borderType": cv2.BORDER_CONSTANT,
            }
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=(0.0, 0.0, 0.0), **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)
            # Random cropping
            base_h, base_w = label.shape
            start_h = random.randint(0, base_h - self.crop_size)
            start_w = random.randint(0, base_w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image, label

    def _load_data(self, image_id):
        # Set paths
        image_path = osp.join(self.root, "images", image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", image_id + ".mat")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        # Load a label map
        if self.version == "1.1":
            label = sio.loadmat(label_path)["S"].astype(np.int64)
            label -= 1  # unlabeled (0 -> -1)
        else:
            label = np.array(h5py.File(label_path, "r")["S"], dtype=np.int64)
            label = label.transpose(1, 0)
            label -= 2  # unlabeled (1 -> -1)
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
        fmt_str += "    Version: {}\n".format(self.version)
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

    dataset_root = "/media/kazuto1011/Extra/cocostuff/cocostuff-10k-v" + "1.1"
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
            mean = (
                torch.tensor((104.008, 116.669, 122.675))
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
            )
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
            # plt.savefig("./docs/data.png", bbox_inches="tight", transparent=True)
            plt.show()
            quit()
