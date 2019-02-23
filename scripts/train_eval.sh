#!/bin/bash

set -x


# 0. Choose from {voc12, cocostuff10k, cocostuff164k}
DATASET=voc12


# 1. Train DeepLab v2 on ${DATASET}
python main.py train \
-c configs/${DATASET}.yaml

# Trained models are saved into
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_5000.pth
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_10000.pth
#   data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_15000.pth
#   ...

# Tensorboard logs are in data/logs.


# 2. Evaluate the model on val set
python main.py test \
-c configs/${DATASET}.yaml \
-m data/models/${DATASET}/deeplabv2_resnet101_msc/*/checkpoint_final.pth

# Validation scores on 4 metrics are saved as
#   data/scores/${DATASET}/deeplabv2_resnet101_msc/*/scores.json

# Logits are saved into
#   data/features/${DATASET}/deeplabv2_resnet101_msc/*/logit/...


# 3. Re-evaluate the model with CRF post-processing
python main.py crf \
-c configs/${DATASET}.yaml

# Scores with CRF on 4 metrics are saved as
#   data/scores/${DATASET}/deeplabv2_resnet101_msc/*/scores_crf.json