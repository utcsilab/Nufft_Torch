# Nufft_Torch
PyTorch Implementation of NUFFT. Allows users to use track gradients for both input images and NUFFT coordinates. Based on SigPy and https://github.com/tomer196/pytorch-nufft with changes to the adjoint to fix overlapping coordinate errors. 

Used to for non-Cartesian motion correction here: https://github.com/utcsilab/motion_score_mri

If you find this repository useful, please consider citing the following paper:
```
@article{levac2022motion,
  title={Accelerated Motion Correction for MRI using Score-Based Generative Models},
  author={Levac, Brett and Jalal, Ajil and Tamir, Jonathan I},
  journal={arXiv preprint arXiv:2211.00199},
  year={2022}
}
```
