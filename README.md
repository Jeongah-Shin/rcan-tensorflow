# rcan-tensorflow
Image Super-Resolution Using Very Deep Residual Channel Attention Networks Implementation in Tensorflow

[ECCV 2018 paper](https://arxiv.org/pdf/1807.02758.pdf)

[Orig PyTorch Implementation](https://github.com/yulunzhang/RCAN)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/kozistr/rcan-tensorflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/kozistr/rcan-tensorflow/context:python)

## Datsets Feeding
```
tfds.core.DatasetInfo(
    name='div2k',
    version=2.0.0,
    description='DIV2K dataset: DIVerse 2K resolution high quality images as used for the challenges @ NTIRE (CVPR 2017 and CVPR 2018) and @ PIRM (ECCV 2018)',
    homepage='https://data.vision.ee.ethz.ch/cvl/DIV2K/',
    features=FeaturesDict({
        'hr': Image(shape=(None, None, 3), dtype=tf.uint8),
        'lr': Image(shape=(None, None, 3), dtype=tf.uint8),
    }),
    total_num_examples=900,
    splits={
        'train': 800,
        'validation': 100,
    },
    supervised_keys=('lr', 'hr'),
    citation="""@InProceedings{Ignatov_2018_ECCV_Workshops,
    author = {Ignatov, Andrey and Timofte, Radu and others},
    title = {PIRM challenge on perceptual image enhancement on smartphones: report},
    booktitle = {European Conference on Computer Vision (ECCV) Workshops},
    url = "http://www.vision.ee.ethz.ch/~timofter/publications/Agustsson-CVPRW-2017.pdf",
    month = {January},
    year = {2019}
    }""",
    redistribution_info=,
)
```

## Introduction
This repo contains my implementation of RCAN (Residual Channel Attention Networks).

Here're the proposed architectures in the paper.

- Channel Attention (CA)
![CA](./assets/channel_attention.png)

- Residual Channel Attention Block (RCAB)
![RCAB](./assets/residual_channel_attention_block.png)

- Residual Channel Attention Network (RCAN), Residual Group (GP)
![RG](./assets/residual_group.png)

**All images got from [the paper](https://arxiv.org/pdf/1807.02758.pdf)**

## Dependencies
* Python
* Tensorflow 1.x
* tqdm
* h5py
* scipy
* cv2

## DataSet

| DataSet |        LR       |        HR        |
|  :---:  |    :-------:    |    :-------:     |
|  DIV2K  |  800 (192x192)  |   800 (768x768)  |
 
## Usage
### training
    # hyper-paramters in config.py, you can edit them!
    $ python3 train.py --data_from [img or h5]
### testing
    $ python3 test.py --src_image sample.png --dst_image sample-upscaled.png

## Results

* OOM on my machine :(... I can't test my code, but maybe code runs fine.

| Example\Resolution | *192x192x3 image (sample)* | *768x768x3 image (generated)* |
|    :-------:       |     :-------:     |     :-------:     |
|  Example1 (X4 scaled)  | ![img](./output/sample_lr.png) | ![img](./output/100000.png) |

## To-Do
1. None

## Author
HyeongChan Kim / [@kozistr](http://kozistr.tech)
