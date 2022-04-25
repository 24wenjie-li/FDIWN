# FDIWN-Pytorch
This repository is an official PyTorch implementation of our paper "Feature Distillation Interaction Weighting Network for Lightweight Image Super-Resolution". (AAAI 2022)

Paper can be download from <a href="https://arxiv.org/abs/2112.08655">FDIWN</a>. 

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
All our SR images can be downloaded from <a href="https://pan.baidu.com/s/1BfATKktSv9jk3LlWPRQRZg">here</a>.[百度网盘][提取码:0824]

All our Supplementary materials can be downloaded from <a href="https://pan.baidu.com/s/1XwdEjCgiPfHTumGU4aWKiQ">here</a>.[百度网盘][提取码:9168]

All pretrained model can be found in <a href="https://github.com/24wenjie-li/FDIWN/tree/main/FDIWN_TrainCode/experiment">AAAI2022_FDIWN_premodel</a>.

The following PSNR/SSIMs are evaluated on Matlab R2017a and the code can be referred to [Evaluate_PSNR_SSIM.m]
(https://github.com/24wenjie-li/FDIWN/blob/main/FDIWN_TestCode/Evaluate_PSNR_SSIM.m).


## Training
Don't use --ext sep argument on your first running.

You can skip the decoding part and use saved binaries with --ext sep argument in second time.
 
```
  cd Train/
  # FDIWN x2  LR: 48 * 48  HR: 96 * 96
  python main.py --model FDIWNx2 --save FDIWNx2 --scale 2 --n_feats 64  --reset --chop --save_results --patch_size 96 --ext sep
  
  # FDIWN x3  LR: 48 * 48  HR: 144 * 144
  python main.py --model FDIWNx3 --save FDIWNx3 --scale 3 --n_feats 64  --reset --chop --save_results --patch_size 144 --ext sep
  
  # FDIWN x4  LR: 48 * 48  HR: 192 * 192
  python main.py --model FDIWNx4 --save FDIWNx4 --scale 4 --n_feats 64  --reset --chop --save_results --patch_size 192 --ext sep
```

## Testing
Using pre-trained model for training, all test datasets must be pretreatment by ''FDIWN_TestCode/Prepare_TestData_HR_LR.m" and all pre-trained model should be put into "FDIWN_TestCode/model/".

```
#FDIWN x2
python main.py --data_test MyImage --scale 2 --model FDIWNx2 --n_feats 64 --pre_train /home/ggw/wenjieli/RCAN/RCAN_TestCode/model/model_best.pt --test_only --save_results --chop --save FDIWNx2 --testpath ../LR/LRBI --testset Set5

#FDIWN x3
python main.py --data_test MyImage --scale 3 --model FDIWNx3 --n_feats 64 --pre_train /home/ggw/wenjieli/RCAN/RCAN_TestCode/model/model_best.pt --test_only --save_results --chop --save FDIWNx3 --testpath ../LR/LRBI --testset Set5

#FDIWN x4
python main.py --data_test MyImage --scale 4 --model FDIWNx4 --n_feats 64 --pre_train /home/ggw/wenjieli/RCAN/RCAN_TestCode/model/model_best.pt --test_only --save_results --chop --save FDIWNx4 --testpath ../LR/LRBI --testset Set5
```

## Performance

Our FDIWN is trained on RGB, but as in previous work, we only reported PSNR/SSIM on the Y channel.

We use  the file  ''...FDIWN_TestCode/Evaluate_PSNR_SSIM'' for test.

Model|Scale|Params|Multi-adds|Set5|Set14|B100|Urban100|Manga109
--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:
FDIWN-M        |x2|433K|73.6G|38.03/0.9606|33.60/0.9179|32.17/0.8995|32.19/0.9284|null/null
FDIWN          |x2|629K|112.0G|38.07/0.9608|33.75/0.9201|32.23/0.9003|32.40/0.9305|38.85/0.9774
FDIWN-M        |x3|446K|35.9G|34.46/0.9274|30.35/0.8423|29.10/0.8051|28.16/0.8528|null/null
FDIWN          |x3|645K|51.5G|34.52/0.9281|30.42/0.8438|29.14/0.8065|28.36/0.8567|33.77/0.9456
FDIWN-M        |x4|454K|19.6G|32.17/0.8941|28.55/0.7806|27.58/0.7364|26.02/0.7844|null/null
FDIWN          |x4|664K|28.4G|32.23/0.8955|28.66/0.7829|27.62/0.7380|26.28/0.7919|30.63/0.9098

<p align="center">
<img src="images/urbanx2_img091.png" width="400px" height="300px"/>
</p>

## Citation

If you find FDIWN useful in your research, please consider citing:
```
@article{gao2021feature,
  title={Feature Distillation Interaction Weighting Network for Lightweight Image Super-Resolution},
  author={Gao, Guangwei and Li, Wenjie and Li, Juncheng and Wu, Fei and Lu, Huimin and Yu, Yi},
  journal={arXiv preprint arXiv:2112.08655},
  year={2021}
}
