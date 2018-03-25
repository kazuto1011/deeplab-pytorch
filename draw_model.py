#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-06

import torch
import yaml
from graphviz import Digraph
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from libs.models import DeepLabV2, DeepLabV2_ResNet101_MSC


def make_dot(var, params):

    node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="30,30"))
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


model = DeepLabV2_ResNet101_MSC(n_classes=183)

input = Variable(torch.randn(1, 3, 513, 513))

# y = model(input)
# g = make_dot(y, model.state_dict())
# g.view(filename='model', cleanup=True)

with SummaryWriter('runs/graph', comment='DeepLabV2') as w:
    w.add_graph(model, (input, ))