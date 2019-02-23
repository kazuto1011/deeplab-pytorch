#!/bin/bash

DATASET_DIR=$1

# Download PASCAL VOC12 (2GB)
wget -nc -P $DATASET_DIR http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract images, annotations, etc.
tar -xvf $DATASET_DIR/VOCtrainval_11-May-2012.tar -C $DATASET_DIR