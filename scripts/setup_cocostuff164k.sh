#!/bin/bash

DATASET_DIR=$1

# Download COCO-Stuff 164k (20GB+)
wget -nc -P $DATASET_DIR http://images.cocodataset.org/zips/train2017.zip
wget -nc -P $DATASET_DIR http://images.cocodataset.org/zips/val2017.zip
wget -nc -P $DATASET_DIR http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

mkdir -p $DATASET_DIR/images
mkdir -p $DATASET_DIR/annotations
unzip -n $DATASET_DIR/train2017.zip -d $DATASET_DIR/images/
unzip -n $DATASET_DIR/val2017.zip -d $DATASET_DIR/images/
unzip -n $DATASET_DIR/stuffthingmaps_trainval2017.zip -d $DATASET_DIR/annotations/

echo ===============================================================================================
echo "Set the path below to \"ROOT:\" in the config/cocostuff164k.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ===============================================================================================