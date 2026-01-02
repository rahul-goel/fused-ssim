import torch
from fused_ssim3d import fused_ssim3d
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt
import numpy as np
import time
import os

plt.style.use('ggplot')

# GPU Device Detection and Configuration
# This script supports benchmarking on multiple GPU backends:
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

if __name__ == "__main__":
    # Benchmark Configuration
    torch.manual_seed(0)

    B, CH = 5, 1
    dimensions = list(range(20, 150, 5))
    iterations = 30

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    pm_ssim = SSIM(data_range=1.0, channel=CH, spatial_dims=3)

    # Training Benchmark: Measure forward and backward pass performance
    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d, d], device=fused_ssim_device)
            img2_og = torch.rand([B, CH, d,d, d], device=fused_ssim_device)

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

        begin = time.time()
        for _ in range(iterations):
            pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            pm_ssim_val.backward()
        fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
        end = time.time()
        data["pytorch_mssim"].append((end - begin) / iterations * 1000)

        begin = time.time()
        for _ in range(iterations):
            mine_ssim_val_same = fused_ssim3d(img1_mine_same, img2_mine_same)
            mine_ssim_val_same.backward()
        fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
        end = time.time()
        data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 3) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one training iteration (ms).")
    plt.title(f"3D Training Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", f"3D_training_time-{gpu.lower().replace(' ', '-')}.png"), dpi=300)

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    plt.clf()

    # Inference Benchmark: Measure forward pass only (no gradients)
    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d, d], device=fused_ssim_device)
            img2_og = torch.rand([B, CH, d, d, d], device=fused_ssim_device)

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

            begin = time.time()
            for _ in range(iterations):
                pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
            end = time.time()
            data["pytorch_mssim"].append((end - begin) / iterations * 1000)

            begin = time.time()
            for _ in range(iterations):
                mine_ssim_val_same = fused_ssim3d(img1_mine_same, img2_mine_same, train=False)
            fused_ssim_module.synchronize()  # Ensure GPU operations complete before timing
            end = time.time()
            data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 3) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one inference iteration (ms).")
    plt.title(f"3D inference Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", f"3D_inference_time-{gpu.lower().replace(' ', '-')}.png"), dpi=300)