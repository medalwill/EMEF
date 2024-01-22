# 🌟 EMEF: Ensemble Multi-Exposure Image Fusion, AAAI 2023 <a href="https://arxiv.org/abs/2305.12734"><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> 
- This repository is the official PyTorch implementation of EMEF
- ✨We will release our code as soon as possible✨

<!-- ![teaser](figures/pipeline.png)
![teaser](figures/maincompare.png) -->
<img src="figures/pipeline.png" width="100%">
<img src="figures/maincompare.png" width="100%">

## Result
- Download our results from [Google Drive](https://drive.google.com/drive/folders/151LaYxeIk9Q0SZS9dVzucWCqiQ4ejhJe?usp=sharing).
- Download our results from [Baidu Netdisk](https://pan.baidu.com/s/1KIHv6sILjqqUO5dBAk3ptA?pwd=emef) (code: emef).

## Requirements
- Install python and [pytorch](https://pytorch.org/get-started/locally/) correctly. 
- Install other requirements.

Here is the example with the help of conda environment:
```
conda create -n emef python=3.10
conda activate emef
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets
TODO

## Training
Start visdom for visualization:
```
python -m visdom.server
```
Start training:
```
python train.py --dataroot {path_to_SICE}/train --name demo --model demo --gpu_ids 1 --display_port 8097
```



## Evaluation
- We use the evaluation code from [MEFB](https://github.com/xingchenzhang/MEFB).
- We use a pytorch implementation of [MEF-SSIM](https://github.com/ChuangbinC/pytorch-MEF_SSIM) to optimize and evaluate our results.

## Acknowledgements
Great thanks for [MEFB](https://github.com/xingchenzhang/MEFB) and all of the open source MEF methods. EMEF can't live without their public available code.

## Citation
If you find our work useful in your research, please cite with:
```
@article{liu2023emef,
  title={EMEF: Ensemble Multi-Exposure Image Fusion},
  volume={37}, 
  url={https://ojs.aaai.org/index.php/AAAI/article/view/25259}, 
  DOI={10.1609/aaai.v37i2.25259}, 
  number={2}, 
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Liu, Renshuai and Li, Chengyang and Cao, Haitao and Zheng, Yinglin and Zeng, Ming and Cheng, Xuan}, 
  year={2023}, 
  month={Jun.}, 
  pages={1710-1718} }
```