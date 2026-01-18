import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import time
from math import exp
sys.path.insert(0, '.')
from fused_ssim_halide import fusedssim

# Reference Implementation
# Taken from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def reference_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    window = window.to(img1.device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# Test sizes matching the C++ tests
TEST_SIZES = [
    (1, 1, 3, "1x1x3"),
    (3, 3, 3, "3x3x3"),
    (15, 15, 3, "15x15x3"),
    (16, 16, 3, "16x16x3"),
    (17, 17, 3, "17x17x3"),
    (64, 64, 3, "64x64x3"),
    (128, 128, 3, "128x128x3"),
    (256, 256, 3, "256x256x3"),
    (512, 512, 3, "512x512x3"),
]

# SSIM Constants
C1 = 0.01 ** 2
C2 = 0.03 ** 2

print("="*70)
print("Halide SSIM vs Reference Implementation Comparison")
print("="*70)
print()

results = []

for width, height, channels, name in TEST_SIZES:
    print(f"Testing size: {name} ({width} x {height} x {channels})")
    print("-" * 70)

    # Create test tensors (NCHW format: batch, channels, height, width)
    img1 = torch.rand(1, channels, height, width, dtype=torch.float32, device="cpu")
    img2 = torch.rand(1, channels, height, width, dtype=torch.float32, device="cpu")

    # Test 1: Halide implementation
    print("  Running Halide SSIM...")
    start = time.perf_counter()

    ssim_map_halide, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, True)

    halide_time = time.perf_counter() - start

    halide_mean = ssim_map_halide.mean().item()
    print(f"    Time: {halide_time*1000:.3f} ms")
    print(f"    Mean SSIM: {halide_mean:.6f}")

    # Test 2: Reference implementation
    ref_time = None
    ref_mean = None
    diff = None

    # Reference requires at least 11x11 images (window size)
    if width >= 11 and height >= 11:
        print("  Running Reference SSIM...")
        start = time.perf_counter()

        ssim_ref = reference_ssim(img1, img2, window_size=11, size_average=True)

        ref_time = time.perf_counter() - start

        ref_mean = ssim_ref.item()
        print(f"    Time: {ref_time*1000:.3f} ms")
        print(f"    Mean SSIM: {ref_mean:.6f}")

        diff = abs(halide_mean - ref_mean)
        speedup = ref_time / halide_time if halide_time > 0 else 0

        print(f"  Difference: {diff:.6e}")
        print(f"  Speedup: {speedup:.2f}x")
    elif width < 11 or height < 11:
        print("  Reference SSIM: Skipped (image too small, requires 11x11 minimum)")

    print()

    results.append({
        'size': name,
        'width': width,
        'height': height,
        'channels': channels,
        'halide_time': halide_time,
        'halide_mean': halide_mean,
        'ref_time': ref_time,
        'ref_mean': ref_mean,
        'diff': diff,
    })

# Summary
print("="*70)
print("SUMMARY")
print("="*70)
print()
print(f"{'Size':<15} {'Halide (ms)':<12} {'Reference (ms)':<15} {'Speedup':<10} {'Diff':<12}")
print("-" * 70)

for r in results:
    halide_ms = r['halide_time'] * 1000
    if r['ref_time'] is not None:
        ref_ms = r['ref_time'] * 1000
        speedup = r['ref_time'] / r['halide_time'] if r['halide_time'] > 0 else 0
        diff_str = f"{r['diff']:.2e}" if r['diff'] is not None else "N/A"
        print(f"{r['size']:<15} {halide_ms:<12.3f} {ref_ms:<15.3f} {speedup:<10.2f}x {diff_str:<12}")
    else:
        print(f"{r['size']:<15} {halide_ms:<12.3f} {'N/A':<15} {'N/A':<10} {'N/A':<12}")

print()

# Test identical images
print("="*70)
print("SANITY CHECK: Identical Images (should be ~1.0)")
print("="*70)
img_test = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cpu")
ssim_identical, _, _, _ = fusedssim(C1, C2, img_test, img_test, True)
print(f"Halide mean SSIM (identical): {ssim_identical.mean().item():.6f}")

ssim_ref_identical = reference_ssim(img_test, img_test, window_size=11, size_average=True)
print(f"Reference mean SSIM (identical): {ssim_ref_identical.item():.6f}")
print(f"Difference: {abs(ssim_identical.mean().item() - ssim_ref_identical.item()):.6e}")

print()
print("="*70)
print("Tests complete!")
print("="*70)
