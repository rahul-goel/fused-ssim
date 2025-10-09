import torch
import numpy as np
import os
from PIL import Image
from fused_ssim import fused_ssim

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

# Load ground truth image and normalize to [0, 1] range
gt_image = torch.tensor(np.array(Image.open(os.path.join("..", "images", "albert.jpg"))), dtype=torch.float32, device=fused_ssim_device).unsqueeze(0).unsqueeze(0) / 255.0

# Initialize predicted image with random values (to be optimized)
pred_image = torch.nn.Parameter(torch.rand_like(gt_image))

# Calculate initial SSIM value
with torch.no_grad():
    ssim_value = fused_ssim(pred_image, gt_image, train=False)
    print("Starting with SSIM value:", ssim_value)

# Setup optimizer for training
optimizer = torch.optim.Adam([pred_image])

# Training loop: Optimize predicted image to match ground truth using SSIM loss
while ssim_value < 0.9999:
    optimizer.zero_grad()
    loss = 1.0 - fused_ssim(pred_image, gt_image)
    loss.backward()
    optimizer.step()

    # Evaluate current SSIM value
    with torch.no_grad():
        ssim_value = fused_ssim(pred_image, gt_image, train=False)
        print("SSIM value:", ssim_value)

# Save the optimized predicted image
pred_image = (pred_image * 255.0).squeeze(0).squeeze(0)
to_save = pred_image.detach().cpu().numpy().astype(np.uint8)

Image.fromarray(to_save).save(os.path.join("..", "images", f"predicted-{gpu.lower().replace(' ', '-')}.jpg"))
