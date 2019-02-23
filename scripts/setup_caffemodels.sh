#!/bin/bash

# Download released caffemodels
wget -nc -P ./data http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip

unzip -n ./data/prototxt_and_model.zip -d ./data

# Move caffemodels to data directories
## MSCOCO
mv ./data/init.caffemodel ./data/models/coco/deeplabv1_resnet101/caffemodel
## PASCAL VOC 2012
mv ./data/train_iter_20000.caffemodel ./data/models/voc12/deeplabv2_resnet101_msc/caffemodel
mv ./data/train2_iter_20000.caffemodel ./data/models/voc12/deeplabv2_resnet101_msc/caffemodel

echo ===============================================================================================
echo "Next, try running script below:"
echo -e "\033[32m python convert.py --dataset coco \033[00m"
echo ===============================================================================================