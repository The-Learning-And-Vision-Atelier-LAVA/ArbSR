# ArbSR
Pytorch implementation of "Learning A Single Network for Scale-Arbitrary Super-Resolution", ICCV 2021


[[Project]](https://longguangwang.github.io/Project/ArbSR/) [[arXiv]](https://arxiv.org/abs/2004.03791)

## Highlights
- ***A plug-in module*** to extend a baseline SR network (e.g., EDSR and RCAN) to a scale-arbitrary SR network with ***small additional computational and memory cost***. 
- Promising results for ***scale-arbitrary SR (both non-integer and asymmetric scale factors)*** while maintaining the state-of-the-art performance for SR with integer scale factors.

## Demo

![gif](./Figs/1.gif)

## Motivation
Although recent CNN-based single image SR networks (e.g., EDSR, RDN and RCAN) have achieved promising performance, they are developed for image SR with a single specific integer scale (e.g., x2, x3, x4). In real-world applications, non-integer SR (e.g., from 100x100 to 220x220) and asymmetric SR (e.g., from 100x100 to 220x420) are also necessary such that customers can zoom in an image arbitrarily for better view of details.

## Overview
![overview](./Figs/overview.png)

## Requirements
- Python 3.6
- PyTorch == 1.1.0
- numpy
- skimage
- imageio
- cv2

## Train
### 1. Prepare training data 

1.1 Download DIV2K training data (800 training images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

1.2 Cd to `./utils` and run `gen_training_data.m` in Matlab to prepare HR/LR images in `your_data_path` as belows:
```
your_data_path
└── DIV2K
	├── HR
		├── 0001.png
		├── ...
		└── 0800.png
	└── LR_bicubic
		├── X1.10
			├── 0001.png
			├── ...
			└── 0800.png
		├── ...
		└── X4.00_X3.50
			├── 0001.png
			├── ...
			└── 0800.png
```

1.3 Specify `--dir_data` based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

### 2. Begin to train
Run `./main.sh` to train on the DIV2K dataset. Please update `dir_data` in the bash file as `your_data_path`.


## Test
### 1. Prepare test data 

1.1 Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets).

1.2 Cd to `./utils` and run `gen_test_data.m` in Matlab to prepare HR/LR images in `your_data_path` as belows:
```
your_data_path
└── benchmark
	├── Set5
		├── HR
			├── baby.png
			├── ...
			└── woman.png
		└── LR_bicubic
			├── X1.10
				├── baby.png
				├── ...
				└── woman.png
			├── ...
			└── X4.00_X3.50
				├── baby.png
				├── ...
				└── woman.png
	├── Set14
	├── B100
	├── Urban100
	└── Manga109
		├── HR
			├── AisazuNihalrarenai.png
			├── ...
			└── YouchienBoueigumi.png
		└── LR_bicubic
			├── X1.10
				├── AisazuNihalrarenai.png
				├── ...
				└── YouchienBoueigumi.png
			├── ...
			└── X4.00_X3.50
				├── AisazuNihalrarenai.png
				├── ...
				└── YouchienBoueigumi.png
```

### 2. Begin to test
Run `./test.sh` to test on benchmark datasets. Please update `dir_data` in the bash file as `your_data_path`.


## Quick Test on An LR Image
Run `./quick_test.sh` to enlarge an LR image to an arbitrary size. Please update `img_dir` in the bash file as `your_img_path`.

## Visual Results
### SR with Symmetric Scale Factors

![non-integer](./Figs/non-integer.png)

### SR with Asymmetric Scale Factors

![asymmetric](./Figs/asymmetric.png)

### SR with Continuous Scale Factors

Please try our [[interactive viewer]](https://longguangwang.github.io/Project/ArbSR/).

## Citation
```
@InProceedings{Wang2020Learning,
  title={Learning A Single Network for Scale-Arbitrary Super-Resolution},
  author={Longguang Wang, Yingqian Wang, Zaiping Lin, Jungang Yang, Wei An, and Yulan Guo},
  booktitle={ICCV},
  year={2021}
}
```

## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [Meta-SR](https://github.com/XuecaiHu/Meta-SR-Pytorch). We thank the authors for sharing the codes.