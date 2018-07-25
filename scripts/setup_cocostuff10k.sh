#!/bin/bash

DATASET_DIR=$1

# Download COCO-Stuff 10k (2GB)
wget -nc -P $DATASET_DIR http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

unzip -n $DATASET_DIR/cocostuff-10k-v1.1.zip -d $DATASET_DIR

echo ===============================================================================================
echo "Set the path below to \"ROOT:\" in the config/cocostuff10k.yaml:"
echo -e "\033[32m $DATASET_DIR \033[00m"
echo ===============================================================================================