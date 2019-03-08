# PASCAL VOC 2012

This is an instruction for setting up PASCAL VOC dataset.

![](../../../docs/datasets/voc12.png)

1. Download PASCAL VOC 2012.

```sh
$ bash scripts/setup_voc12.sh [PATH TO DOWNLOAD]
```

```
/VOCdevkit
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   └── Segmentation
    ├── JPEGImages
    ├── SegmentationObject
    └── SegmentationClass
```

2. Add SBD augmentated training data as `SegmentationClassAug`.


* Convert by yourself ([here](https://github.com/shelhamer/fcn.berkeleyvision.org/tree/master/data/pascal)).
* Or download pre-converted files ([here](https://github.com/DrSleep/tensorflow-deeplab-resnet#evaluation)).

3. Download official image sets as `ImageSets/SegmentationAug`.

* From https://ucla.app.box.com/s/rd9z2xvwsfpksi7mi08i2xqrj7ab4keb/file/55053033642
* Or https://github.com/kazuto1011/deeplab-pytorch/files/2945588/list.zip

```sh
/VOCdevkit
└── VOC2012
    ├── Annotations
    ├── ImageSets
    │   ├── Segmentation
    │   └── SegmentationAug # ADDED!!
    │       ├── test.txt
    │       ├── train_aug.txt
    │       ├── train.txt
    │       ├── trainval_aug.txt
    │       ├── trainval.txt
    │       └── val.txt
    ├── JPEGImages
    ├── SegmentationObject
    ├── SegmentationClass
    └── SegmentationClassAug # ADDED!!
        └── 2007_000032.png
```

1. Set the path to the dataset in ```configs/voc12.yaml```.

```yaml
DATASET: voc12
    ROOT: # <- Write here
...
```
