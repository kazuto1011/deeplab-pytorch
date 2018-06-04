#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from __future__ import absolute_import, division, print_function

import re
from collections import OrderedDict

import click
import numpy as np
import torch

from libs import caffe_pb2
from libs.models import DeepLabV2_ResNet101_MSC


def parse_caffemodel(model_path):
    caffemodel = caffe_pb2.NetParameter()
    with open(model_path, "rb") as f:
        caffemodel.MergeFromString(f.read())

    # Check trainable layers
    print(
        *set([(layer.type, len(layer.blobs)) for layer in caffemodel.layer]), sep="\n"
    )

    params = OrderedDict()
    previous_layer_type = None
    for layer in caffemodel.layer:
        print("{} ({}): {}".format(layer.name, layer.type, len(layer.blobs)))

        # Skip the shared branch
        if "res075" in layer.name or "res05" in layer.name:
            continue

        # Convolution or Dilated Convolution
        if "Convolution" in layer.type:
            params[layer.name] = {}
            params[layer.name]["kernel_size"] = layer.convolution_param.kernel_size[0]
            params[layer.name]["weight"] = list(layer.blobs[0].data)
            if len(layer.blobs) == 2:
                params[layer.name]["bias"] = list(layer.blobs[1].data)
            if len(layer.convolution_param.stride) == 1:  # or []
                params[layer.name]["stride"] = layer.convolution_param.stride[0]
            else:
                params[layer.name]["stride"] = 1
            if len(layer.convolution_param.pad) == 1:  # or []
                params[layer.name]["padding"] = layer.convolution_param.pad[0]
            else:
                params[layer.name]["padding"] = 0
            if isinstance(layer.convolution_param.dilation, int):
                params[layer.name]["dilation"] = layer.convolution_param.dilation
            elif len(layer.convolution_param.dilation) == 1:
                params[layer.name]["dilation"] = layer.convolution_param.dilation[0]
            else:
                params[layer.name]["dilation"] = 1
        # Batch Normalization
        elif "BatchNorm" in layer.type:
            params[layer.name] = {}
            params[layer.name]["running_mean"] = (
                np.array(layer.blobs[0].data) / layer.blobs[2].data[0]
            )
            params[layer.name]["running_var"] = (
                np.array(layer.blobs[1].data) / layer.blobs[2].data[0]
            )
            params[layer.name]["eps"] = layer.batch_norm_param.eps
            params[layer.name][
                "momentum"
            ] = layer.batch_norm_param.moving_average_fraction
            batch_norm_layer = layer.name
        # Scale
        elif "Scale" in layer.type:
            assert previous_layer_type == "BatchNorm"
            params[batch_norm_layer]["weight"] = list(layer.blobs[0].data)
            params[batch_norm_layer]["bias"] = list(layer.blobs[1].data)

        previous_layer_type = layer.type

    return params


# Hard coded translater
def translate_layer_name(source):
    def layer_block_branch(source, target):
        target += ".layer{}".format(source[0][0])
        if len(source[0][1:]) == 1:
            block = {"a": 1, "b": 2, "c": 3}.get(source[0][1:])
        else:
            block = int(source[0][2:]) + 1
        target += ".block{}".format(block)
        branch = source[1][6:]
        if branch == "1":
            target += ".proj"
        elif branch == "2a":
            target += ".reduce"
        elif branch == "2b":
            target += ".conv3x3"
        elif branch == "2c":
            target += ".increase"
        return target

    source = source.split("_")
    target = "scale"

    if "conv1" in source[0]:
        target += ".layer1.conv1.conv"
    elif "conv1" in source[1]:
        target += ".layer1.conv1.bn"
    elif "res" in source[0]:
        source[0] = source[0].replace("res", "")
        target = layer_block_branch(source, target)
        target += ".conv"
    elif "bn" in source[0]:
        source[0] = source[0].replace("bn", "")
        target = layer_block_branch(source, target)
        target += ".bn"
    elif "fc" in source[0]:
        # Skip if coco_init
        if len(source) == 3:
            stage = source[2]
            target += ".aspp.stages.{}".format(stage)

    return target


@click.command()
@click.option(
    "--dataset", required=True, type=click.Choice(["voc12", "coco_init", "init"])
)
def main(dataset):
    WHITELIST = ["kernel_size", "stride", "padding", "dilation", "eps", "momentum"]
    CONFIG = {
        "voc12": {
            "path_caffe_model": "data/models/deeplab_resnet101/voc12/train2_iter_20000.caffemodel",
            "path_pytorch_model": "data/models/deeplab_resnet101/voc12/deeplabv2_resnet101_VOC2012.pth",
            "n_classes": 21,
        },
        "coco_init": {
            "path_caffe_model": "data/models/deeplab_resnet101/coco_init/init.caffemodel",
            "path_pytorch_model": "data/models/deeplab_resnet101/coco_init/deeplabv2_resnet101_COCO_init.pth",
            "n_classes": 91,
        },
        "init": {
            # The same as the coco_init parameters
            "path_caffe_model": "data/models/deeplab_resnet101/init/deeplabv2_resnet101_init.caffemodel",
            "path_pytorch_model": "data/models/deeplab_resnet101/init/deeplabv2_resnet101_init.pth",
            "n_classes": 91,
        },
    }.get(dataset)

    params = parse_caffemodel(CONFIG["path_caffe_model"])

    model = DeepLabV2_ResNet101_MSC(n_classes=CONFIG["n_classes"])
    model.eval()
    own_state = model.state_dict()

    state_dict = OrderedDict()
    for layer_name, layer_dict in params.items():
        for param_name, values in layer_dict.items():
            if param_name in WHITELIST and dataset != "coco_init" and dataset != "init":
                attribute = translate_layer_name(layer_name)
                attribute = eval("model." + attribute + "." + param_name)
                if isinstance(attribute, tuple):
                    if attribute[0] != values:
                        raise ValueError
                else:
                    if abs(attribute - values) > 1e-4:
                        raise ValueError
                print(
                    layer_name.ljust(20),
                    "->",
                    param_name,
                    attribute,
                    values,
                    ": Checked!",
                )
                continue
            param_name = translate_layer_name(layer_name) + "." + param_name
            if param_name in own_state:
                values = torch.FloatTensor(values)
                values = values.view_as(own_state[param_name])
                state_dict[param_name] = values
                print(layer_name.ljust(20), "->", param_name, ": Copied!")

    torch.save(state_dict, CONFIG["path_pytorch_model"])


if __name__ == "__main__":
    main()
