import torch
from fused_ssim import fused_ssim
from pytorch_msssim import SSIM
import matplotlib.pyplot as plt
import numpy as np
import time
import os

plt.style.use('ggplot')

if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name()
    fused_ssim_device = "cuda"
    fused_ssim_module = torch.cuda
elif torch.mps.is_available():
    gpu = "Apple Silicon (MPS)"
    fused_ssim_device = "mps"
    fused_ssim_module = torch.mps

if __name__ == "__main__":
    torch.manual_seed(0)

    B, CH = 5, 1
    dimensions = list(range(50, 1550, 50))
    iterations = 50

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    pm_ssim = SSIM(data_range=1.0, channel=CH)

    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d], device=fused_ssim_device)
            img2_og = torch.rand([B, CH, d, d], device=fused_ssim_device)

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

        begin = time.time()
        for _ in range(iterations):
            pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            pm_ssim_val.backward()
        fused_ssim_module.synchronize()
        end = time.time()
        data["pytorch_mssim"].append((end - begin) / iterations * 1000)

        begin = time.time()
        for _ in range(iterations):
            mine_ssim_val_same = fused_ssim(img1_mine_same, img2_mine_same)
            mine_ssim_val_same.backward()
        fused_ssim_module.synchronize()
        end = time.time()
        data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 2) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one training iteration (ms).")
    plt.title(f"Training Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", f"training_time_{fused_ssim_device}.png"), dpi=300)

    data = {
        "pytorch_mssim": [],
        "fused-ssim": []
    }

    plt.clf()
    for d in dimensions:
        with torch.no_grad():
            img1_og = torch.rand([B, CH, d, d], device=fused_ssim_device)
            img2_og = torch.rand([B, CH, d, d], device=fused_ssim_device)

            img1_mine_same = torch.nn.Parameter(img1_og.clone())
            img2_mine_same = img2_og.clone()

            img1_pm = torch.nn.Parameter(img1_og.clone())
            img2_pm = img2_og.clone()

            begin = time.time()
            for _ in range(iterations):
                pm_ssim_val = pm_ssim(img1_pm, img2_pm)
            fused_ssim_module.synchronize()
            end = time.time()
            data["pytorch_mssim"].append((end - begin) / iterations * 1000)

            begin = time.time()
            for _ in range(iterations):
                mine_ssim_val_same = fused_ssim(img1_mine_same, img2_mine_same, train=False)
            fused_ssim_module.synchronize()
            end = time.time()
            data["fused-ssim"].append((end - begin) / iterations * 1000)

    num_pixels = (B * np.array(dimensions) ** 2) / 1e6
    plt.plot(num_pixels, data["pytorch_mssim"], label="pytorch_mssim")
    plt.plot(num_pixels, data["fused-ssim"], label="fused-ssim")
    plt.legend()
    plt.xlabel("Number of pixels (in millions).")
    plt.ylabel("Time for one inference iteration (ms).")
    plt.title(f"Inference Benchmark on {gpu}.")
    plt.savefig(os.path.join("..", "images", f"inference_time_{fused_ssim_device}.png"), dpi=300)
