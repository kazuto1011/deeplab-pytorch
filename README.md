# DeepLab with PyTorch

This is an unofficial **PyTorch** implementation of **DeepLab v2** [[1](##references)] with a **ResNet-101** backbone. 
* **COCO-Stuff** dataset [[2](##references)] and **PASCAL VOC** dataset [[3]()] are supported.
* The official Caffe weights provided by the authors can be used without building the Caffe APIs.
* DeepLab v3/v3+ models with the identical backbone are also included (not tested).
* [```torch.hub``` is supported](#torchhub).

## Performance

### COCO-Stuff

<table>
    <tr>
        <th>Train set</th>
        <th>Eval set</th>
        <th>Code</th>
        <th>Weight</th>
        <th>CRF?</th>
        <th>Pixel<br>Accuracy</th>
        <th>Mean<br>Accuracy</th>
        <th>Mean IoU</th>
        <th>FreqW IoU</th>
    </tr>
    <tr>
        <td rowspan="3">
            10k <i>train</i> &dagger;
        </td>
        <td rowspan="3">10k <i>val</i> &dagger;</td>
        <td>Official [<a href="#references">2</a>]</td>
        <td></td>
        <td></td>
        <td><strong>65.1</strong></td>
        <td><strong>45.5</strong></td>
        <td><strong>34.4</strong></td>
        <td><strong>50.4</strong></td>
    </tr>
    <tr>
        <td rowspan="2"><strong>This repo</strong></td>
        <td rowspan="2"><a href="https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-cocostuff10k-20000.pth">Download</a></td>
        <td></td>
        <td><strong>65.8</td>
        <td><strong>45.7</strong></td>
        <td><strong>34.8</strong></td>
        <td><strong>51.2</strong></td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>67.1</td>
        <td>46.4</td>
        <td>35.6</td>
        <td>52.5</td>
    </tr>
    <tr>
        <td rowspan="2">
            164k <i>train</i>
        </td>
        <td rowspan="2">164k <i>val</i></td>
        <td rowspan="2"><strong>This repo</strong></td>
        <td rowspan="2"><a href="https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-cocostuff164k-100000.pth">Download</a> &Dagger;</td>
        <td></td>
        <td>66.8</td>
        <td>51.2</td>
        <td>39.1</td>
        <td>51.5</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>67.6</td>
        <td>51.5</td>
        <td>39.7</td>
        <td>52.3</td>
    </tr>
</table>

&dagger; Images and labels are pre-warped to square-shape 513x513<br>
&Dagger; Note for [SPADE](https://nvlabs.github.io/SPADE/) followers: The provided COCO-Stuff 164k weight has been kept intact since 2019/02/23.

### PASCAL VOC 2012

<table>
    <tr>
        <th>Train set</th>
        <th>Eval set</th>
        <th>Code</th>
        <th>Weight</th>
        <th>CRF?</th>
        <th>Pixel<br>Accuracy</th>
        <th>Mean<br>Accuracy</th>
        <th>Mean IoU</th>
        <th>FreqW IoU</th>
    </tr>
    <tr>
        <td rowspan="4">
            <i>trainaug</i>
        </td>
        <td rowspan="4"><i>val</i></td>
        <td rowspan="2">Official [<a href="#references">3</a>]</td>
        <td rowspan="2"></td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td><strong>76.35</strong></td>
        <td>-</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>-</td>
        <td>-</td>
        <td><strong>77.69</strong></td>
        <td>-</td>
    </tr>
    <tr>
        <td rowspan="2"><strong>This repo</strong></td>
        <td rowspan="2"><a href="https://github.com/kazuto1011/deeplab-pytorch/releases/download/v1.0/deeplabv2_resnet101_msc-vocaug-20000.pth">Download</a></td>
        <td></td>
        <td>94.64</td>
        <td>86.50</td>
        <td><strong>76.65</td>
        <td>90.41</td>
    </tr>
    <tr>
        <td>&#10003;</td>
        <td>95.04</td>
        <td>86.64</td>
        <td><strong>77.93</strong></td>
        <td>91.06</td>
    </tr>
</table>

## Setup

### Requirements

Required Python packages are listed in the Anaconda configuration file `configs/conda_env.yaml`.
Please modify the listed `cudatoolkit=10.2` and `python=3.6` as needed and run the following commands.

```sh
# Set up with Anaconda
conda env create -f configs/conda_env.yaml
conda activate deeplab-pytorch
```

### Download datasets

* [COCO-Stuff 10k/164k](data/datasets/cocostuff/README.md)
* [PASCAL VOC 2012](data/datasets/voc12/README.md)

### Download pre-trained caffemodels

Caffemodels pre-trained on COCO and PASCAL VOC datasets are released by the DeepLab authors.
In accordance with the papers [[1](##references),[2](##references)], this repository uses the COCO-trained parameters as initial weights.

1. Run the follwing script to download the pre-trained caffemodels (1GB+).

```sh
$ bash scripts/setup_caffemodels.sh
```

2. Convert the caffemodels to pytorch compatibles. No need to build the Caffe API!

```sh
# Generate "deeplabv1_resnet101-coco.pth" from "init.caffemodel"
$ python convert.py --dataset coco
# Generate "deeplabv2_resnet101_msc-vocaug.pth" from "train2_iter_20000.caffemodel"
$ python convert.py --dataset voc12
```

## Training & Evaluation

To train DeepLab v2 on PASCAL VOC 2012:

```sh
python main.py train \
    --config-path configs/voc12.yaml
```

To evaluate the performance on a validation set:

```sh
python main.py test \
    --config-path configs/voc12.yaml \
    --model-path data/models/voc12/deeplabv2_resnet101_msc/train_aug/checkpoint_final.pth
```

Note: This command saves the predicted logit maps (`.npy`) and the scores (`.json`).

To re-evaluate with a CRF post-processing:<br>

```sh
python main.py crf \
    --config-path configs/voc12.yaml
```

Execution of a series of the above scripts is equivalent to `bash scripts/train_eval.sh`.

To monitor a loss, run the following command in a separate terminal.

```sh
tensorboard --logdir data/logs
```

Please specify the appropriate configuration files for the other datasets.

| Dataset         | Config file                  | #Iterations | Classes                      |
| :-------------- | :--------------------------- | :---------- | :--------------------------- |
| PASCAL VOC 2012 | `configs/voc12.yaml`         | 20,000      | 20 foreground + 1 background |
| COCO-Stuff 10k  | `configs/cocostuff10k.yaml`  | 20,000      | 182 thing/stuff              |
| COCO-Stuff 164k | `configs/cocostuff164k.yaml` | 100,000     | 182 thing/stuff              |

Note: Although the label indices range from 0 to 181 in COCO-Stuff 10k/164k, only [171 classes](https://github.com/nightrome/cocostuff/blob/master/labels.md) are supervised.

Common settings:

- **Model**: DeepLab v2 with ResNet-101 backbone. Dilated rates of ASPP are (6, 12, 18, 24). Output stride is 8.
- **GPU**: All the GPUs visible to the process are used. Please specify the scope with
```CUDA_VISIBLE_DEVICES=```.
- **Multi-scale loss**: Loss is defined as a sum of responses from multi-scale inputs (1x, 0.75x, 0.5x) and element-wise max across the scales. The *unlabeled* class is ignored in the loss computation.
- **Gradient accumulation**: The mini-batch of 10 samples is not processed at once due to the high occupancy of GPU memories. Instead, gradients of small batches of 5 samples are accumulated for 2 iterations, and weight updating is performed at the end (```batch_size * iter_size = 10```). GPU memory usage is approx. 11.2 GB with the default setting (tested on the single Titan X). You can reduce it with a small ```batch_size```.
- **Learning rate**: Stochastic gradient descent (SGD) is used with momentum of 0.9 and initial learning rate of 2.5e-4. Polynomial learning rate decay is employed; the learning rate is multiplied by ```(1-iter/iter_max)**power``` at every 10 iterations.
- **Monitoring**: Moving average loss (```average_loss``` in Caffe) can be monitored in TensorBoard.
- **Preprocessing**: Input images are randomly re-scaled by factors ranging from 0.5 to 1.5, padded if needed, and randomly cropped to 321x321.

Processed images and labels in COCO-Stuff 164k:

![Data](docs/datasets/cocostuff.png)

## Inference Demo

You can use [the pre-trained models](#performance), [the converted models](#download-pre-trained-caffemodels), or your models.

To process a single image:

```bash
python demo.py single \
    --config-path configs/voc12.yaml \
    --model-path deeplabv2_resnet101_msc-vocaug-20000.pth \
    --image-path image.jpg
```

To run on a webcam:

```bash
python demo.py live \
    --config-path configs/voc12.yaml \
    --model-path deeplabv2_resnet101_msc-vocaug-20000.pth
```

To run a CRF post-processing, add `--crf`. To run on a CPU, add `--cpu`.

## Misc

### torch.hub

Model setup with two lines

```python
import torch.hub
model = torch.hub.load("kazuto1011/deeplab-pytorch", "deeplabv2_resnet101", pretrained='cocostuff164k', n_classes=182)
```

### Difference with Caffe version

* While the official code employs 1/16 bilinear interpolation (```Interp``` layer) for downsampling a label for only 0.5x input, this codebase does for both 0.5x and 0.75x inputs with nearest interpolation (```PIL.Image.resize```, [related issue](https://github.com/kazuto1011/deeplab-pytorch/issues/51)).
* Bilinear interpolation on images and logits is performed with the ```align_corners=False```.

### Training batch normalization


This codebase only supports DeepLab v2 training which freezes batch normalization layers, although
v3/v3+ protocols require training them. If training their parameters on multiple GPUs as well in your projects, please
install [the extra library](https://hangzhang.org/PyTorch-Encoding/) below.

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

1. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, A. L. Yuille. DeepLab: Semantic Image
Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs. *IEEE TPAMI*,
2018.<br>
[Project](http://liangchiehchen.com/projects/DeepLab.html) /
[Code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) / [arXiv
paper](https://arxiv.org/abs/1606.00915)

2. H. Caesar, J. Uijlings, V. Ferrari. COCO-Stuff: Thing and Stuff Classes in Context. In *CVPR*, 2018.<br>
[Project](https://github.com/nightrome/cocostuff) / [arXiv paper](https://arxiv.org/abs/1612.03716)

1. M. Everingham, L. Van Gool, C. K. I. Williams, J. Winn, A. Zisserman. The PASCAL Visual Object
Classes (VOC) Challenge. *IJCV*, 2010.<br>
[Project](http://host.robots.ox.ac.uk/pascal/VOC) /
[Paper](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)
