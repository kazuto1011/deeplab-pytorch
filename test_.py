import os
import sys
from collections import OrderedDict
from os import walk

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import DeepLab_ResNet
import torchvision.models as models
from docopt import docopt

docstr = """Evaluate ResNet-DeepLab trained on scenes (VOC 2012),a total of 21 labels including background

Usage:
    evalpyt.py [options]

Options:
    -h, --help                  Print this message
    --visualize                 view outputs of each sketch
    --snapPrefix=<str>          Snapshot [default: VOC12_scenes_]
    --testGTpath=<str>          Ground truth path prefix [default: data/gt/]
    --testIMpath=<str>          Sketch images path prefix [default: data/img/]
    --NoLabels=<int>            The number of different labels in training data, VOC has 21 labels, including background [default: 21]
    --gpu0=<int>                GPU number [default: 0]
"""

args = docopt(docstr, version='v0.1')
print args


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print 'pred shape', pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = int(args['--NoLabels']) - 1  # labels from 0,1, ... 20(for VOC)
    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(), x[1].tolist()))
        # pdb.set_trace()
        n_jj = set.intersection(p_idx_j, GT_idx_j)
        u_jj = set.union(p_idx_j, GT_idx_j)

        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:]) / float(len(np.unique(gt)))

    return Aiou


gpu0 = int(args['--gpu0'])
im_path = args['--testIMpath']
model = DeepLab_ResNet(int(args['--NoLabels']))
model.eval()
counter = 0
model.cuda(gpu0)
snapPrefix = args['--snapPrefix']
gt_path = args['--testGTpath']
img_list = open('data/list/val.txt').readlines()

# TODO set the (different iteration)models that you want to evaluate on. Models are saved during training after every 1000 iters by default.
for iter in range(1, 21):
    saved_state_dict = torch.load(os.path.join(
        'data/snapshots/', snapPrefix + str(iter) + '000.pth'))
    if counter == 0:
        print snapPrefix
    counter += 1
    model.load_state_dict(saved_state_dict)

    pytorch_list = []
    for i in img_list:
        img = np.zeros((513, 513, 3))

        img_temp = cv2.imread(os.path.join(
            im_path, i[:-1] + '.jpg')).astype(float)
        img_original = img_temp
        img_temp[:, :, 0] = img_temp[:, :, 0] - 104.008
        img_temp[:, :, 1] = img_temp[:, :, 1] - 116.669
        img_temp[:, :, 2] = img_temp[:, :, 2] - 122.675
        img[:img_temp.shape[0], :img_temp.shape[1], :] = img_temp
        gt = cv2.imread(os.path.join(gt_path, i[:-1] + '.png'), 0)
        gt[gt == 255] = 0

        output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(
            0, 3, 1, 2)).float(), volatile=True).cuda(gpu0))
        interp = nn.UpsamplingBilinear2d(size=(513, 513))
        output = interp(output[3]).cpu().data[0].numpy()
        output = output[:, :img_temp.shape[0], :img_temp.shape[1]]

        output = output.transpose(1, 2, 0)
        output = np.argmax(output, axis=2)
        if args['--visualize']:
            plt.subplot(3, 1, 1)
            plt.imshow(img_original)
            plt.subplot(3, 1, 2)
            plt.imshow(gt)
            plt.subplot(3, 1, 3)
            plt.imshow(output)
            plt.show()

        iou_pytorch = get_iou(output, gt)
        pytorch_list.append(iou_pytorch)

    print 'pytorch', iter, np.sum(np.asarray(pytorch_list)) / len(pytorch_list)
