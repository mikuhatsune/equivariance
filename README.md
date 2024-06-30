## Improving Equivariance in State-of-the-Art Supervised Depth and Normal Predictors
Yuanyi Zhong, Anand Bhattad, Yu-Xiong Wang, David Forsyth. ICCV 2023.
https://arxiv.org/abs/2309.16646
```
@inproceedings{zhong2023improving,
  title={Improving Equivariance in State-of-the-Art Supervised Depth and Normal Predictors},
  author={Zhong, Yuanyi and Bhattad, Anand and Wang, Yu-Xiong and Forsyth, David},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={21775--21785},
  year={2023}
}
```

Example:
* Taskonomy depth prediction, run on 4 GPUs, num crops: k=3, equivariance loss coefficient: equi_coef=1e-4 imposed on the last_conv1 layer of a UNet
```bash
torchrun --nproc_per_node=4 main.py --dev=0,1,2,3 --tag=k3_eq1e-4_last_conv1 --k=3 --equi_coef=1e-4 --task=depth --equi_layers=last_conv1
```