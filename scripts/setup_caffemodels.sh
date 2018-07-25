#!/bin/bash

# Download released caffemodels
wget -nc http://liangchiehchen.com/projects/released/deeplab_aspp_resnet101/prototxt_and_model.zip

unzip -n prototxt_and_model.zip

# Move caffemodels to data directories
## MSCOCO
mv init.caffemodel data/models/deeplab_resnet101/coco_init
## PASCAL VOC 2012
mv train_iter_20000.caffemodel data/models/deeplab_resnet101/voc12
mv train2_iter_20000.caffemodel data/models/deeplab_resnet101/voc12

# Remove *.prototxt
rm *.prototxt

echo ===============================================================================================
echo "Next, try running script below:"
echo -e "\033[32m python convert.py --dataset coco_init \033[00m"
echo ===============================================================================================