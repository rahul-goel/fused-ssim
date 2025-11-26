# 3D Fully Fused Differentiable SSIM

This repository contains an adaptation of [fused-ssim](https://github.com/rahul-goel/fused-ssim) for PyTorch that also supports 3D images. It is up to 8x faster in 2D and up to 11x faster in 3D compared to the previous fastest implementation - [pytorch-msssim](https://github.com/VainF/pytorch-msssim). 

The 3D implementation (```fused_ssim3d```) retains the same assumptions as 2D, following the original SSIM paper and translating the 2D optimizations to 3D. It is also possible to call the original 2D instance of fused-ssim (now under ```fused_ssim2d```).

Check out the original [fused-ssim](https://github.com/rahul-goel/fused-ssim) repository for further background and details why this approach is faster. 

## Hardware Compatibility

Only NVIDIA GPUs are supported. The support for the other hardware was not carried over from 2D fused-ssim as of now.

- **NVIDIA GPUs** (CUDA).

- ~~**AMD GPUs** (ROCm).~~ ~~**Apple Silicon** (Metal Performance Shaders).~~
 ~~**Intel GPUs** (SYCL).~~

## Installation Instructions

### Prerequisites

You must have PyTorch installed with the appropriate backend for your GPU before installing fused-ssim. The installation process requires the backend compilers to be available.

### Step 1: Install PyTorch with CUDA

First, ensure you have CUDA Toolkit installed on your system (version 11.8 or 12.x recommended).

```bash
# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify NVCC (CUDA compiler) is available:
```bash
nvcc --version
```


### Step 2: Install Fused-SSIM3D

Once PyTorch and the appropriate backend compiler are installed:

```bash
# Install from GitHub (recommended)
pip install git+https://github.com/PaPieta/fused-ssim3D --no-build-isolation

# Or clone and install locally
git clone https://github.com/PaPieta/fused-ssim3D
cd fused-ssim3D
pip install . --no-build-isolation
```

The setup.py script will automatically detect your GPU architecture. For verbose output:

```bash
pip install git+https://github.com/rahul-goel/fused-ssim/ -v --no-build-isolation
```

If the above commands don't work, try:

```bash
python setup.py install
```

### Troubleshooting

- **CUDA errors**: Ensure your CUDA Toolkit version matches your PyTorch CUDA version

## Usage

### 3D
```python
import torch
from fused_ssim3d import fused_ssim3d

# predicted_image, gt_image: [BS, CH, D, H, W]
# predicted_image is differentiable
gt_image = torch.rand(2, 3, 96, 96, 96)
predicted_image = torch.nn.Parameter(torch.rand_like(gt_image))
ssim_value = fused_ssim3d(predicted_image, gt_image)
```

### 2D (legacy)
```python
import torch
from fused_ssim3d import fused_ssim2d

# predicted_image, gt_image: [BS, CH, H, W]
# predicted_image is differentiable
gt_image = torch.rand(2, 3, 1080, 1920)
predicted_image = torch.nn.Parameter(torch.rand_like(gt_image))
ssim_value = fused_ssim2d(predicted_image, gt_image)
```

By default, `same` padding is used. To use `valid` padding which is the kind of padding used by [pytorch-mssim](https://github.com/VainF/pytorch-msssim):
```python
ssim_value = fused_ssim3d(predicted_image, gt_image, padding="valid")
```

If you don't want to train and use this only for inference, use the following for even faster speed:
```python
with torch.no_grad():
  ssim_value = fused_ssim3d(predicted_image, gt_image, train=False)
```

## Constraints

### Legacy
- Currently, only one of the images is allowed to be differentiable i.e. only the first image can be `nn.Parameter`.
- Images must be normalized to range `[0, 1]`.
- Standard `11x11` convolutions supported.

## Performance
The performance of the 2D version is retained (see [here](https://github.com/rahul-goel/fused-ssim/tree/main?tab=readme-ov-file#performance)). In 3D we reach up to 11x acceleration, which can mostly be accredited to the poor performance of the baseline [pytorch-msssim](https://github.com/VainF/pytorch-msssim) in 3D. For the same number of pixels, ```fused_ssim3d``` is still slower than its 2D counterpart.

<img src="./images/3D_training_time-nvidia-rtx-a5000.png" width="45%"> <img src="./images/3D_inference_time-nvidia-rtx-a5000.png" width="45%"> 

## BibTeX
If you use this 3D fused SSIM implementation in your work, please cite both the original paper and this repository:
```
@misc{fusedssim3d2025,
    title        = {{3D} Fully Fused Differentiable {SSIM}},
    author       = {Pawel Tomasz Pieta},
    year         = 2025,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/PaPieta/fused-ssim3D}}
}
@inproceedings{taming3dgs,
    author = {Mallick, Saswat Subhajyoti and Goel, Rahul and Kerbl, Bernhard and Steinberger, Markus and Carrasco, Francisco Vicente and De La Torre, Fernando},
    title = {Taming 3DGS: High-Quality Radiance Fields with Limited Resources},
    year = {2024},
    url = {https://doi.org/10.1145/3680528.3687694},
    doi = {10.1145/3680528.3687694},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
    series = {SA '24}
}
```

## 3D Implementation details

A natural extension of the 2D fused-ssim would be to load a 3D neighbourhood into shared memory (```sTile```). While on paper the fastest, this method is strongly limited by shared memory size constraints. It is possible to make it work by heavy reuse of the a single large array (see e.g [this kernel version](https://github.com/PaPieta/fused-ssim3D/blob/609a2f185be620cd4bda634b377c5c75f0af9b74/ssim3d.cu)), but the shared memory is still big enough to only enable a handful (often 1-3) blocks per SM, which in turn greatly affects latency hiding during memory access. With this approach, the acceleration was ~5x.

An alternative approach (implemented here) is to:
1. Perform 2D Gaussian convolutions on each slice (similarly to the ssim2d architecture),
2. Save intermediate results to global memory
3. Make each thread calculate the Z axis convolution and final SSIM along the whole Z row (depth).

The last part is achieved by loading appropriate data from global into a ring buffer. With each step, only a missing data point is loaded and the "center pixel" index adjusted.