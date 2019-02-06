# DeepLab with PyTorch <!-- omit in toc -->

This is an unofficial **PyTorch** implementation to train **DeepLab v2** model (ResNet backbone) [[1](##references)] on **COCO-Stuff** dataset [[2](##references)].
DeepLab is one of the CNN architectures for semantic image segmentation.
COCO-Stuff is a semantic segmentation dataset, which includes 164k images annotated with 171 thing/stuff classes (+ unlabeled).

This repository aims to reproduce the official score of DeepLab v2 on COCO-Stuff datasets.
The model can be trained both on the latest [COCO-Stuff 164k](https://github.com/nightrome/cocostuff) and [COCO-Stuff 10k](https://github.com/nightrome/cocostuff10k), *without building the official implementation in Caffe*.
[Pre-trained models are provided](#models----omit-in-toc).
ResNet-based DeepLab v3/v3+ are also included, although they are not tested.
```torch.hub``` is supported.

- [Setup](#setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Performance](#performance)
- [Demo](#demo)
- [Misc](#misc)
- [References](#references)

## Setup

### Requirements <!-- omit in toc -->

For anaconda users:

```sh
# Please modify CUDA option according to your environment
conda env create --file config/conda_env.yaml
```

* python 2.7+/3.6+
* [pytorch](https://pytorch.org/) 0.4.1+
* [torchvision](https://pytorch.org/)
* [torchnet](https://github.com/pytorch/tnt)
* [pydensecrf](https://github.com/lucasb-eyer/pydensecrf)
* [tensorflow](https://www.tensorflow.org/install/) (for tensorboard)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) 1.0+
* opencv 3.0.0+
* tqdm
* click
* addict
* h5py
* scipy
* matplotlib
* yaml
* joblib

### Datasets <!-- omit in toc -->

COCO-Stuff 164k is the latest version and recommended.

<details>
<summary><strong>COCO-Stuff 164k</strong> (click to show the structure)</summary>
<pre>
├── images
│   ├── train2017
│   │   ├── 000000000009.jpg
│   │   └── ...
│   └── val2017
│       ├── 000000000139.jpg
│       └── ...
└── annotations
    ├── train2017
    │   ├── 000000000009.png
    │   └── ...
    └── val2017
        ├── 000000000139.png
        └── ...
</pre>
</details>
<br>

1. Run the script below to download the dataset (20GB+).

```sh
./scripts/setup_cocostuff164k.sh <PATH TO DOWNLOAD>
```

2. Set the path to the dataset in ```config/cocostuff164k.yaml```.

```yaml
DATASET: cocostuff164k
ROOT: # <- Write here
...
```

<details>
<summary><strong>COCO-Stuff 10k</strong> (click to show the structure)</summary>
<pre>
├── images
│   ├── COCO_train2014_000000000077.jpg
│   └── ...
├── annotations
│   ├── COCO_train2014_000000000077.mat
│   └── ...
└── imageLists
    ├── all.txt
    ├── test.txt
    └── train.txt
</pre>
</details>
<br>

1. Run the script below to download the dataset (2GB).

```sh
./scripts/setup_cocostuff10k.sh <PATH TO DOWNLOAD>
```

2. Set the path to the dataset in ```config/cocostuff10k.yaml```.

```yaml
DATASET: cocostuff10k
ROOT: # <- Write here
...
```

### Initial parameters <!-- omit in toc -->

1. Run the script below to download caffemodel pre-trained on ImageNet and 91-class COCO (1GB+).

```sh
./scripts/setup_caffemodels.sh
```

2. Convert the caffemodel to pytorch compatible. No need to build the official DeepLab!

```sh
# This generates deeplabv2_resnet101_COCO_init.pth from init.caffemodel
python convert.py --dataset init
```
You can also convert an included ```train2_iter_20000.caffemodel``` for PASCAL VOC 2012 dataset. See [here](config/README.md#voc12yaml). 

## Training

Training, evaluation, and some demos are all through the [```.yaml``` configuration files](config/README.md).

```sh
# Train DeepLab v2 on COCO-Stuff 164k
python main.py train --config config/cocostuff164k.yaml
```

```sh
# Monitor a cross-entropy loss
tensorboard --logdir runs
```

Default settings:

- All the GPUs visible to the process are used. Please specify the scope with ```CUDA_VISIBLE_DEVICES=```.
- Stochastic gradient descent (SGD) is used with momentum of 0.9 and initial learning rate of 2.5e-4. Polynomial learning rate decay is employed; the learning rate is multiplied by ```(1-iter/iter_max)**power``` at every 10 iterations.
- Weights are updated 20k iterations for COCO-Stuff 10k and 100k iterations for COCO-Stuff 164k, with a mini-batch of 10. The batch is not processed at once due to high occupancy of video memories, instead, gradients of small batches are aggregated, and weight updating is performed at the end (```batch_size * iter_size = 10```).
- Input images are initially warped to 513x513, randomly re-scaled by factors ranging from 0.5 to 1.5, zero-padded if needed, and randomly cropped to 321x321 so that the input size is fixed during training (see the example below).
- The label indices range from 0 to 181 and the model outputs a 182-dim categorical distribution, but only [171 classes](https://github.com/nightrome/cocostuff/blob/master/labels.md) are supervised with COCO-Stuff.
- Loss is defined as a sum of responses from multi-scale inputs (1x, 0.75x, 0.5x) and element-wise max across the scales. The "unlabeled" class (index -1) is ignored in the loss computation.
- Moving average loss (```average_loss``` in Caffe) can be monitored in TensorBoard.
- GPU memory usage is approx. 11.2 GB with the default setting (tested on the single Titan X). You can reduce it with a small ```batch_size```.

Processed image vs. label examples:

![Data](docs/data.png)

## Evaluation

```sh
# Evaluate the final model on COCO-Stuff 164k validation set
python main.py test --config config/cocostuff164k.yaml \
                    --model-path data/models/deeplab_resnet101/cocostuff164k/checkpoint_final.pth
```

You can run CRF post-processing with a option ```--crf```. See ```--help``` for more details.

## Performance

### Validation scores <!-- omit in toc -->

| &nbsp;                                                            |   Train set   |  Eval set   |  CRF?  | Pixel<br>Accuracy | Mean<br>Accuracy |  Mean IoU  | FreqW IoU  |
| :---------------------------------------------------------------- | :-----------: | :---------: | :----: | :---------------: | :--------------: | :--------: | :--------: |
| [**Official (Caffe)**](https://github.com/nightrome/cocostuff10k) | **10k train** | **10k val** | **No** |    **65.1 %**     |    **45.5 %**    | **34.4 %** | **50.4 %** |
| **This repo**                                                     | **10k train** | **10k val** | **No** |    **65.3 %**     |    **45.3 %**    | **34.4 %** | **50.5 %** |
| This repo                                                         |   10k train   |   10k val   |  Yes   |      66.7 %       |      45.9 %      |   35.5 %   |   51.9 %   |
| This repo                                                         |  164k train   |   10k val   |   No   |      67.6 %       |      54.9 %      |   43.2 %   |   53.9 %   |
| This repo                                                         |  164k train   |   10k val   |  Yes   |      68.7 %       |      55.3 %      |   44.4 %   |   55.1 %   |
| This repo                                                         |  164k train   |  164k val   |   No   |      65.7 %       |      49.7 %      |   37.6 %   |   50.0 %   |
| This repo                                                         |  164k train   |  164k val   |  Yes   |      66.8 %       |      50.1 %      |   38.5 %   |   51.1 %   |

### Models <!-- omit in toc -->

* [Models trained on COCO-Stuff 10k/164k (*.pth)](https://drive.google.com/drive/folders/1m3wyXvvWy-IvGmdFS_dsQCRXhFNhek8_?usp=sharing)
* [Scores (*.json)](https://drive.google.com/drive/folders/1PouglnlwsyHTwdSo_d55WgMgdnxbxmE6?usp=sharing)

## Demo

### From an image <!-- omit in toc -->

```bash
python demo.py single --config config/cocostuff164k.yaml \
                      --model-path <PATH TO MODEL> \
                      --image-path <PATH TO IMAGE> \
                      --crf
```

### From a webcam <!-- omit in toc -->

A class of mouseovered pixel is shown in terminal.

```bash
python demo.py live --config config/cocostuff164k.yaml \
                    --model-path <PATH TO MODEL> \
                    --camera-id <CAMERA ID> \
                    --crf
```

### torch.hub <!-- omit in toc -->

Model setup with 3 lines.

```python
import torch.hub

model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", n_classes=182)
model.load_state_dict(torch.load("cocostuff164k_iter100k.pth"))
```

## Misc

### Image processing <!-- omit in toc -->

Default setting warps an image and a label to square-shape as the official code does.
To preserve aspect ratio in the image preprocessing, please modify ```.yaml``` as follows:

```yaml
BATCH_SIZE:
    TEST: 1
WARP_IMAGE: False
```

### Training batch normalization <!-- omit in toc -->

This codebase only supports DeepLab v2 training which freezes batch normalization layers, although v3/v3+ protocols require training them. If training their parameters as well in your projects, please install [the extra library](https://hangzhang.org/PyTorch-Encoding/) below.

```bash
pip install torch-encoding
```

Batch normalization layers in a model are automatically switched in ```libs/models/resnet.py```.

```python
try:
    from encoding.nn import SyncBatchNorm
    _BATCH_NORM = SyncBatchNorm
except:
    _BATCH_NORM = nn.BatchNorm2d
```

## References

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*, 2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) / [Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [Code](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)