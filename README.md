# FDIWN-Pytorch
This repository is an official PyTorch implementation of our paper "Feature Distillation Interaction Weighting Network for Lightweight Image Super-Resolution". (AAAI 2022)

## Paper can be download from <a href="https://arxiv.org/abs/2112.08655">FDIWN</a>. 

All test datasets (Preprocessed HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DcrVSz0Q19ySBxiTs4oB">here</a>.

All original test datasets (HR images) can be downloaded from <a href="https://www.jianguoyun.com/p/DaSU0L4Q19ySBxi_qJAB">here</a>.


## Prerequisites:
1. Python 3.6
2. PyTorch >= 0.4.0
3. numpy
4. skimage
5. imageio
6. matplotlib
7. tqdm


## Dataset

We used DIV2K dataset to train our model. Please download it from <a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">here</a>.

Extract the file and put it into the Train/dataset.

Only DIV2K is used as the training dataset, and Flickr2K is not used as the training dataset !!!


## Results
All our SR images can be downloaded from <a href="https://pan.baidu.com/s/1BfATKktSv9jk3LlWPRQRZg">here</a>.[百度网盘][提取码：0824]

All pretrained model can be found in <a href="https://github.com/24wenjie-li/FDIWN/tree/main/FDIWN_TrainCode/experiment">AAAI2022_FDIWN_premodel</a>.

The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to [Evaluate_PSNR_SSIM.m]
(https://github.com/24wenjie-li/FDIWN/blob/main/FDIWN_TestCode/Evaluate_PSNR_SSIM.m).
