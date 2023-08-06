## Contextual Point Cloud Modeling for Weakly-supervised Point Cloud Semantic Segmentation (ICCV 2023)

<p align="center">
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/ICCV-2023-blue.svg">
  </a>
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/License-MIT-green.svg">
  </a>
</p>

Created by [Lizhao Liu](https://scholar.google.com/citations?user=_AMTrAQAAAAJ&hl=zh-CN), Xunlong Xiao, [Zhuangwei Zhuang](https://scholar.google.com/citations?user=T2aPuoYAAAAJ&hl=zh-CN) from the South China University of Technology.

This repository contains the official PyTorch implementation of our ICCV 2023 paper [*Contextual Point Cloud Modeling for Weakly-supervised Point Cloud Semantic Segmentation*](https://arxiv.org/pdf/2307.10316.pdf).

<br>

<img src="figs/CPCM_overview.png" align="center">


### Environment Setup
Our codebase is based on [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), a high performance sparse convolution library built on PyTorch.

We recommend to use MinkowskiEngine 0.5.4, since it is much faster than 0.4.3

For MinkowskiEngine 0.5.4, see instruction in [me054](prepare_env/me054/README.md)

For MinkowskiEngine 0.4.3, see instruction in [me043](prepare_env/me043/README.md)

### Data Preparation
We perform experiments on the following dataset

- [ScanNet V2](https://kaldir.vc.in.tum.de/scannet_benchmark/)
- [S3DIS](http://buildingparser.stanford.edu/dataset.html)
- [Semantic-KITTI (Front view that contains both RGB and XYZ)](http://www.semantic-kitti.org/)

The preprocessed datasets are shared via google drive

- [ScanNet V2 (6.88G)](https://drive.google.com/file/d/16y5f16RI-X-9q7k1_nG9tDb1aqmmrNQt/view?usp=drive_link)
- [S3DIS (8.82G)](https://drive.google.com/file/d/1wD04uB5znFIcY0fY-7U8Ig3jlSX8uczX/view?usp=drive_link)
- [Semantic-KITTI (3.15G)](https://drive.google.com/file/d/1pxScBQrk5uLrDDoKGgOQ4fcq5yMMrCxX/view?usp=drive_link)

Or see instruction in [Dataset Preparation Hand-by-hand](prepare_dataset/README.md) to prepare by yourself.

### S3DIS
To reproduce experiment results of S3DIS

See experiment scripts [here](scripts/S3DIS.sh) for details.

### ScanNet V2

To reproduce experiment results of ScanNet V2

See experiment scripts [here](scripts/ScanNetV2.sh) for details.
