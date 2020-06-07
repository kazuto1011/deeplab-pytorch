#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-15

from __future__ import absolute_import, division, print_function

import re
import traceback
from collections import Counter, OrderedDict

import click
import numpy as np
import torch
from addict import Dict

from libs import caffe_pb2
from libs.models import DeepLabV1_ResNet101, DeepLabV2_ResNet101_MSC


def parse_caffemodel(model_path):
    caffemodel = caffe_pb2.NetParameter()
    with open(model_path, "rb") as f:
        caffemodel.MergeFromString(f.read())

    # Check trainable layers
    print(
        *Counter(
            [(layer.type, len(layer.blobs)) for layer in caffemodel.layer]
        ).most_common(),
        sep="\n",
    )

    params = OrderedDict()
    previous_layer_type = None
    for layer in caffemodel.layer:
        # Skip the shared branch
        if "res075" in layer.name or "res05" in layer.name:
            continue

        print(
            "\033[34m[Caffe]\033[00m",
            "{} ({}): {}".format(layer.name, layer.type, len(layer.blobs)),
        )

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
        # Fully-connected
        elif "InnerProduct" in layer.type:
            params[layer.name] = {}
            params[layer.name]["weight"] = list(layer.blobs[0].data)
            if len(layer.blobs) == 2:
                params[layer.name]["bias"] = list(layer.blobs[1].data)
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
            params[layer.name]["momentum"] = (
                1 - layer.batch_norm_param.moving_average_fraction
            )
            params[layer.name]["num_batches_tracked"] = np.array(0)
            batch_norm_layer = layer.name
        # Scale
        elif "Scale" in layer.type:
            assert previous_layer_type == "BatchNorm"
            params[batch_norm_layer]["weight"] = list(layer.blobs[0].data)
            params[batch_norm_layer]["bias"] = list(layer.blobs[1].data)
        elif "Pooling" in layer.type:
            params[layer.name] = {}
            params[layer.name]["kernel_size"] = layer.pooling_param.kernel_size
            params[layer.name]["stride"] = layer.pooling_param.stride
            params[layer.name]["padding"] = layer.pooling_param.pad

        previous_layer_type = layer.type

    return params


# Hard coded translater
def translate_layer_name(source, target="base"):
    def layer_block_branch(source, target):
        target += "layer{}".format(source[0][0])
        if len(source[0][1:]) == 1:
            block = {"a": 1, "b": 2, "c": 3}.get(source[0][1:])
        else:
            block = int(source[0][2:]) + 1
        target += ".block{}".format(block)
        branch = source[1][6:]
        if branch == "1":
            target += ".shortcut"
        elif branch == "2a":
            target += ".reduce"
        elif branch == "2b":
            target += ".conv3x3"
        elif branch == "2c":
            target += ".increase"
        return target

    source = source.split("_")

    if "pool" in source[0]:
        target += "layer1.pool"
    elif "fc" in source[0]:
        if len(source) == 3:
            stage = source[2]
            target += "aspp.{}".format(stage)
        else:
            target += "fc"
    elif "conv1" in source[0]:
        target += "layer1.conv1.conv"
    elif "conv1" in source[1]:
        target += "layer1.conv1.bn"
    elif "res" in source[0]:
        source[0] = source[0].replace("res", "")
        target = layer_block_branch(source, target)
        target += ".conv"
    elif "bn" in source[0]:
        source[0] = source[0].replace("bn", "")
        target = layer_block_branch(source, target)
        target += ".bn"

    return target


@click.command()
@click.option(
    "-d",
    "--dataset",
    type=click.Choice(["voc12", "coco"]),
    required=True,
    help="Caffemodel",
)
def main(dataset):
    """
    Convert caffemodels to pytorch models
    """

    WHITELIST = ["kernel_size", "stride", "padding", "dilation", "eps", "momentum"]
    CONFIG = Dict(
        {
            "voc12": {
                # For loading the provided VOC 2012 caffemodel
                "PATH_CAFFE_MODEL": "data/models/voc12/deeplabv2_resnet101_msc/caffemodel/train2_iter_20000.caffemodel",
                "PATH_PYTORCH_MODEL": "data/models/voc12/deeplabv2_resnet101_msc/caffemodel/deeplabv2_resnet101_msc-vocaug.pth",
                "N_CLASSES": 21,
                "MODEL": "DeepLabV2_ResNet101_MSC",
                "HEAD": "base.",
            },
            "coco": {
                # For loading the provided initial weights pre-trained on COCO
                "PATH_CAFFE_MODEL": "data/models/coco/deeplabv1_resnet101/caffemodel/init.caffemodel",
                "PATH_PYTORCH_MODEL": "data/models/coco/deeplabv1_resnet101/caffemodel/deeplabv1_resnet101-coco.pth",
                "N_CLASSES": 91,
                "MODEL": "DeepLabV1_ResNet101",
                "HEAD": "",
            },
        }.get(dataset)
    )

    params = parse_caffemodel(CONFIG.PATH_CAFFE_MODEL)

    model = eval(CONFIG.MODEL)(n_classes=CONFIG.N_CLASSES)
    model.eval()
    reference_state_dict = model.state_dict()

    rel_tol = 1e-7

    converted_state_dict = OrderedDict()
    for caffe_layer, caffe_layer_dict in params.items():
        for param_name, caffe_values in caffe_layer_dict.items():
            pytorch_layer = translate_layer_name(caffe_layer, CONFIG.HEAD)
            if pytorch_layer:
                pytorch_param = pytorch_layer + "." + param_name

                # Parameter check
                if param_name in WHITELIST:
                    pytorch_values = eval("model." + pytorch_param)
                    if isinstance(pytorch_values, tuple):
                        assert (
                            pytorch_values[0] == caffe_values
                        ), "Inconsistent values: {} @{} (Caffe), {} @{} (PyTorch)".format(
                            caffe_values,
                            caffe_layer + "/" + param_name,
                            pytorch_values,
                            pytorch_param,
                        )
                    else:
                        assert (
                            abs(pytorch_values - caffe_values) < rel_tol
                        ), "Inconsistent values: {} @{} (Caffe), {} @{} (PyTorch)".format(
                            caffe_values,
                            caffe_layer + "/" + param_name,
                            pytorch_values,
                            pytorch_param,
                        )
                    print(
                        "\033[34m[Passed!]\033[00m",
                        (caffe_layer + "/" + param_name).ljust(35),
                        "->",
                        pytorch_param,
                    )
                    continue

                # Weight conversion
                if pytorch_param in reference_state_dict:
                    caffe_values = torch.tensor(caffe_values)
                    caffe_values = caffe_values.view_as(
                        reference_state_dict[pytorch_param]
                    )
                    converted_state_dict[pytorch_param] = caffe_values
                    print(
                        "\033[32m[Copied!]\033[00m",
                        (caffe_layer + "/" + param_name).ljust(35),
                        "->",
                        pytorch_param,
                    )

    print("\033[32mVerify the converted model\033[00m")
    model.load_state_dict(converted_state_dict)

    print('Saving to "{}"'.format(CONFIG.PATH_PYTORCH_MODEL))
    torch.save(converted_state_dict, CONFIG.PATH_PYTORCH_MODEL)


if __name__ == "__main__":
    main()
