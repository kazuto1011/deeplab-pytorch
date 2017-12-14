#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-06

import argparse

import torch
import yaml
from graphviz import Digraph
from torch.autograd import Variable

from libs.models import *


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """

    param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                dot.node(str(id(var)), size_to_str(u.size()), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__.replace('Backward', '')))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)

    return dot


parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='config/default.yaml')
parser.add_argument('--dataset', type=str, default='cocostuff')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)

image_size = config[args.dataset]['image']['size']['test']
# model = DeepLabV2(n_classes=config[args.dataset]['n_classes'],
#                   n_blocks=[3, 4, 23, 3],
#                   pyramids=[6, 12, 18, 24])
model = PSPNet(n_classes=config[args.dataset]['n_classes'],
               n_blocks=[3, 4, 6, 3],
               pyramids=[6, 3, 2, 1])
model.eval()
y = model(Variable(torch.randn(1, 3, image_size, image_size)))
g = make_dot(y, model.state_dict())

g.view(filename='model', cleanup=True)
