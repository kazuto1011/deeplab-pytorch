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
from glob import glob

import cv2
import h5py
import numpy as np
import scipy.io as sio
import torch
from torch.utils import data
from tqdm import tqdm


class _CocoStuff(data.Dataset):
    """COCO-Stuff base class"""

    def __init__(
        self,
        root,
        split="train",
        base_size=513,
        crop_size=321,
        mean=(104.008, 116.669, 122.675),
        scale=(0.5, 0.75, 1.0, 1.25, 1.5),
        warp=True,
        flip=True,
        preload=False,
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

        self.files = []
        self.images = []
        self.labels = []
        self.ignore_label = None

        self._set_files()

        if self.preload:
            self._preload_data()

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError()

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
            scale_factor = random.choice(self.scale)
            scale_kwargs = {"dsize": None, "fx": scale_factor, "fy": scale_factor}
            image = cv2.resize(image, interpolation=cv2.INTER_LINEAR, **scale_kwargs)
            label = cv2.resize(label, interpolation=cv2.INTER_NEAREST, **scale_kwargs)
            scale_h, scale_w = label.shape
            # Padding
            pad_h = max(max(base_size[1], self.crop_size) - scale_h, 0)
            pad_w = max(max(base_size[0], self.crop_size) - scale_w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
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
        raise NotImplementedError()

    def _preload_data(self):
        for image_id in tqdm(
            self.files, desc="Preloading...", leave=False, dynamic_ncols=True
        ):
            image, label = self._load_data(image_id)
            self.images.append(image)
            self.labels.append(label)

    def __getitem__(self, index):
        if self.preload:
            image, label = self.images[index], self.labels[index]
        else:
            image_id = self.files[index]
            image, label = self._load_data(image_id)
        image, label = self._transform(image, label)
        return image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root Location: {}\n".format(self.root)
        return fmt_str


class CocoStuff10k(_CocoStuff):
    """COCO-Stuff 10k dataset"""

    def __init__(self, version="1.1", **kwargs):
        self.version = version
        self.ignore_label = -1
        super(CocoStuff10k, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list via {train, test, all}.txt
        if self.split in ["train", "test", "all"]:
            file_list = osp.join(self.root, "imageLists", self.split + ".txt")
            file_list = tuple(open(file_list, "r"))
            file_list = [id_.rstrip() for id_ in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

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
        elif self.version == "1.0":
            label = np.array(h5py.File(label_path, "r")["S"], dtype=np.int64)
            label = label.transpose(1, 0)
            label -= 2  # unlabeled (1 -> -1)
        else:
            raise NotImplementedError(
                "1.0 or 1.1 expected, but got: {}".format(self.version)
            )
        return image, label


class CocoStuff164k(_CocoStuff):
    """COCO-Stuff 164k dataset"""

    def __init__(self, **kwargs):
        self.ignore_label = 255
        super(CocoStuff164k, self).__init__(**kwargs)

    def _set_files(self):
        # Create data list by parsing the "images" folder
        if self.split in ["train2017", "val2017"]:
            file_list = sorted(glob(osp.join(self.root, "images", self.split, "*.jpg")))
            assert len(file_list) > 0, "{} has no image".format(
                osp.join(self.root, "images", self.split)
            )
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            self.files = file_list
        else:
            raise ValueError("Invalid split name: {}".format(self.split))

    def _load_data(self, image_id):
        # Set paths
        image_path = osp.join(self.root, "images", self.split, image_id + ".jpg")
        label_path = osp.join(self.root, "annotations", self.split, image_id + ".png")
        # Load an image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE).astype(np.int64)
        return image, label


def get_parent_class(value, dictionary):
    # Get parent class with COCO-Stuff hierarchy
    for k, v in dictionary.items():
        if isinstance(v, list):
            if value in v:
                yield k
        elif isinstance(v, dict):
            if value in list(v.keys()):
                yield k
            else:
                for res in get_parent_class(value, v):
                    yield res


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torchvision
    from torchvision.utils import make_grid
    import yaml

    kwargs = {"nrow": 10, "padding": 50}
    batch_size = 100

    dataset_root = "/media/kazuto1011/Extra/cocostuff/cocostuff-164k"
    dataset = CocoStuff164k(root=dataset_root, split="train2017")

    print(dataset)

    loader = data.DataLoader(dataset, batch_size=batch_size)

    for i, (images, labels) in tqdm(
        enumerate(loader), total=np.ceil(len(dataset) / batch_size), leave=False
    ):

        if i == 0:
            mean = (
                torch.tensor((104.008, 116.669, 122.675))
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
            )
            images += mean.expand_as(images)
            image = make_grid(images, pad_value=-1, **kwargs).numpy()
            image = np.transpose(image, (1, 2, 0))
            mask = np.zeros(image.shape[:2])
            mask[(image != -1)[..., 0]] = 255
            image = np.dstack((image, mask)).astype(np.uint8)

            labels = labels[:, np.newaxis, ...]
            label = make_grid(labels, pad_value=255, **kwargs).numpy()
            label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
            label = cm.jet_r(label_ / 182.0) * 255
            mask = np.zeros(label.shape[:2])
            label[..., 3][(label_ == 255)] = 0
            label = label.astype(np.uint8)

            tiled_images = np.hstack((image, label))
            # cv2.imwrite("./docs/data.png", tiled_images)
            plt.imshow(np.dstack((tiled_images[..., 2::-1], tiled_images[..., 3])))
            plt.show()
            break

    class_hierarchy = "./data/datasets/cocostuff/cocostuff_hierarchy.yaml"
    data = yaml.load(open(class_hierarchy))
    key = "person"

    for _ in range(3):
        key = get_parent_class(key, data)
        key = list(key)[0]
        print(key)
