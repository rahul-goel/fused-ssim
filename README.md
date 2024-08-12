# Fully Fused Differentiable SSIM

This repository contains an efficient fully-fused implementation of [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure) which is differentiable in nature. There are several factors that contribute to an efficient implementation:
- Convolutions in SSIM are spatially localized leading to fully-fused implementation without touching global memory for intermediate steps.
- Backpropagation through Gaussian Convolution is simply another Gaussian Convolution itself.
- Gaussian Convolutions are separable leading to reduced computation.

As per the original SSIM paper, this implementation uses `11x11` sized convolution kernel. The weights for it have been hardcoded and this is another reason for it's speed. This implementation currently only supports **2D images** but with **variable number of channels** and **batch size**.
