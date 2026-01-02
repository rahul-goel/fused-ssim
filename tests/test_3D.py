import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import exp
from fused_ssim import fused_ssim3d
from pytorch_msssim import SSIM
import time

# GPU Device Detection and Configuration
# This script supports multiple GPU backends:
# - CUDA: For Nvidia GPUs and AMD GPUs (via ROCm)
# - MPS: For Apple Silicon (M1, M2, M3, etc.)
# - XPU: For Intel GPUs (via SYCL/oneAPI)
# The appropriate device is automatically detected and configured below.

if torch.cuda.is_available():
    # CUDA backend for Nvidia GPUs and AMD GPUs (with ROCm)
    gpu = torch.cuda.get_device_name()
    fused_ssim_device = "cuda"
    fused_ssim_module = torch.cuda
elif torch.mps.is_available():
    # MPS (Metal Performance Shaders) backend for Apple Silicon
    gpu = "Apple Silicon (MPS)"
    fused_ssim_device = "mps"
    fused_ssim_module = torch.mps
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    # XPU backend for Intel GPUs (via SYCL/oneAPI)
    gpu = torch.xpu.get_device_name(0)
    fused_ssim_device = "xpu"
    fused_ssim_module = torch.xpu

# Reference Implementation is taken from the following:
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/loss_utils.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5)
    _3D_window = (
        _1D_window[:, None, None]
        * _1D_window[None, :, None]
        * _1D_window[None, None, :]
    ).float()
    window = Variable(
        _3D_window.unsqueeze(0)
        .unsqueeze(0)
        .expand(channel, 1, window_size, window_size, window_size)
        .contiguous()
    )
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)

    window = window.to(fused_ssim_device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, CH, D, H, W = 2, 1, 96, 96, 96
    pm_ssim = SSIM(data_range=1.0, channel=CH, spatial_dims=3)
    iterations = 10

    # Correctness Tests: Verify fused-ssim matches reference implementations
    for _ in range(iterations):
        with torch.no_grad():
            img1_og = nn.Parameter(
                torch.rand([B, CH, D, H, W], device=fused_ssim_device, dtype=torch.double)
            )
            img2_og = torch.rand([B, CH, D, H, W], device=fused_ssim_device, dtype=torch.double)

            img1_pm = nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

            img1_mine_same = nn.Parameter(img1_og.clone().to(dtype=torch.float32))
            img2_mine_same = img2_og.clone().to(dtype=torch.float32)

            img1_mine_valid = nn.Parameter(img1_og.clone().to(dtype=torch.float32))
            img2_mine_valid = img2_og.clone().to(dtype=torch.float32)

        og_ssim_val = ssim(img1_og, img2_og)
        mine_ssim_val_same = fused_ssim3d(img1_mine_same, img2_mine_same).double()
        mine_ssim_val_valid = fused_ssim3d(img1_mine_valid, img2_mine_valid, "valid").double()
        pm_ssim_val = pm_ssim(img1_pm, img2_pm)

        assert torch.allclose(og_ssim_val, mine_ssim_val_same, rtol=1e-6, atol=1e-8)
        assert torch.allclose(mine_ssim_val_valid, pm_ssim_val, rtol=1e-6, atol=1e-8)

        og_ssim_val.backward()
        mine_ssim_val_same.backward()
        mine_ssim_val_valid.backward()
        pm_ssim_val.backward()

        assert torch.allclose(
            img1_og.grad,
            img1_mine_same.grad.to(dtype=img1_og.grad.dtype),
            rtol=1e-6,
            atol=1e-8,
        )
        assert torch.allclose(
            img1_mine_valid.grad.to(dtype=img1_pm.grad.dtype),
            img1_pm.grad,
            rtol=1e-6,
            atol=1e-8,
        )

    # Performance Benchmarks: Compare forward and backward pass timings
    img1 = nn.Parameter(torch.rand([B, CH, D, H, W], device=fused_ssim_device))
    img2 = torch.rand([B, CH, D, H, W], device=fused_ssim_device)

    # Benchmark reference implementation
    begin = time.time()
    for _ in range(iterations):
        og_ssim_val = ssim(img1, img2)
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    og_time_forward = (end - begin) / iterations * 1000
    print("Reference Time (Forward):", og_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        og_ssim_val = ssim(img1, img2)
        og_ssim_val.backward()
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    og_time_backward = (end - begin) / iterations * 1000 - og_time_forward
    print("Reference Time (Backward):", og_time_backward, "ms")

    # Benchmark pytorch_mssim
    begin = time.time()
    for _ in range(iterations):
        pm_ssim_val = pm_ssim(img1, img2)
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    pm_time_forward = (end - begin) / iterations * 1000
    print("pytorch_mssim Time (Forward):", pm_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        pm_ssim_val = pm_ssim(img1, img2)
        pm_ssim_val.backward()
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    pm_time_backward = (end - begin) / iterations * 1000 - pm_time_forward
    print("pytorch_mssim Time (Backward):", pm_time_backward, "ms")

    # Benchmark fused-ssim
    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim3d(img1, img2)
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    mine_time_forward = (end - begin) / iterations * 1000
    print("fused-ssim Time (Forward):", mine_time_forward, "ms")

    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim3d(img1, img2)
        mine_ssim_val.backward()
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    mine_time_backward = (end - begin) / iterations * 1000 - mine_time_forward
    print("fused-ssim Time (Backward):", mine_time_backward, "ms")

    begin = time.time()
    for _ in range(iterations):
        mine_ssim_val = fused_ssim3d(img1, img2, train=False)
    fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
    end = time.time()
    mine_time_infer = (end - begin) / iterations * 1000
    print("fused-ssim Time (Inference):", mine_time_infer, "ms")
