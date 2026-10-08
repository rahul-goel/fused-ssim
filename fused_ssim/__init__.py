from typing import NamedTuple
import torch.nn as nn
import torch

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward, fusedssim3d, fusedssim_backward3d
    from fused_ssim_cuda import fusedssim_sum, fusedssim_backward_mean
    is_3D_supported = True
elif torch.mps.is_available():
    from fused_ssim_mps import fusedssim, fusedssim_backward
    is_3D_supported = False
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    from fused_ssim_xpu import fusedssim, fusedssim_backward
    is_3D_supported = False


allowed_padding = ["same", "valid"]

    
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True, spatial_dims=2):
        if spatial_dims == 2:
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)
        elif spatial_dims == 3:
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim3d(C1, C2, img1, img2, train)
        
        if padding == "valid":
            if spatial_dims == 2:
                ssim_map = ssim_map[:, :, 5:-5, 5:-5]
            elif spatial_dims == 3:
                ssim_map = ssim_map[:, :, 5:-5, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        ctx.spatial_dims = spatial_dims

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            if ctx.spatial_dims == 2:
                dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
            elif ctx.spatial_dims == 3:
                dL_dmap[:, :, 5:-5, 5:-5, 5:-5] = opt_grad
        if ctx.spatial_dims == 2:
            grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        elif ctx.spatial_dims == 3:
            grad = fusedssim_backward3d(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)

        return None, None, grad, None, None, None, None

def fused_ssim3d(img1, img2, padding="same", train=True):
    if not is_3D_supported:
        raise RuntimeError("3D fused SSIM is not supported on this device.")
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 3)
    return map.mean()

class FusedSSIMMean(torch.autograd.Function):
    """mean(SSIM map) over the valid region without materializing the map; the
    backward uses dL/dmap = g / count directly."""

    @staticmethod
    def forward(ctx, C1, C2, img1, img2, pad):
        map_sum, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim_sum(C1, C2, img1, img2, pad, False)
        B, CH, H, W = img1.shape
        count = B * CH * max(H - 2 * pad, 0) * max(W - 2 * pad, 0)
        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1, ctx.C2, ctx.pad, ctx.count = C1, C2, pad, count
        return (map_sum / count).squeeze(0)

    @staticmethod
    def backward(ctx, grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        dimg1 = fusedssim_backward_mean(ctx.C1, ctx.C2, img1, img2, grad.reshape(1), float(ctx.count), ctx.pad,
                                        dm_dmu1, dm_dsigma1_sq, dm_dsigma12, None, 0.0)
        return None, None, dimg1, None, None


def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    # Training: reduce the mean inside the kernel (fixed-order, deterministic) instead of writing the map.
    if train and img1.is_cuda and img1.requires_grad:
        if not img1.is_contiguous(memory_format=torch.channels_last):
            img1 = img1.contiguous()
        return FusedSSIMMean.apply(C1, C2, img1, img2, 5 if padding == "valid" else 0)

    # Only the CUDA/HIP 2D kernels read channels-last (e.g. a permuted NHWC render) in place.
    # The Metal and SYCL kernels index flat NCHW buffers.
    if not img1.is_cuda or not img1.is_contiguous(memory_format=torch.channels_last):
        img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 2)
    return map.mean()


class FusedL1SSIMMean(torch.autograd.Function):
    """(mean |img1 - img2|, mean SSIM over the valid region) in one pass; the
    backward adds both image gradients inside the SSIM backward kernel."""

    @staticmethod
    def forward(ctx, C1, C2, img1, img2, pad):
        sums, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim_sum(C1, C2, img1, img2, pad, True)
        B, CH, H, W = img1.shape
        count = B * CH * max(H - 2 * pad, 0) * max(W - 2 * pad, 0)
        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1, ctx.C2, ctx.pad, ctx.count, ctx.l1_count = C1, C2, pad, count, img1.numel()
        return sums[1] / img1.numel(), sums[0] / count

    @staticmethod
    def backward(ctx, g_l1, g_ssim):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        if g_ssim is None:
            g_ssim = torch.zeros((), device=img1.device)
        dimg1 = fusedssim_backward_mean(ctx.C1, ctx.C2, img1, img2, g_ssim.reshape(1), float(ctx.count), ctx.pad,
                                        dm_dmu1, dm_dsigma1_sq, dm_dsigma12,
                                        None if g_l1 is None else g_l1.reshape(1), float(ctx.l1_count))
        return None, None, dimg1, None, None


def fused_l1_ssim(img1, img2, padding="valid"):
    """Returns (mean L1, mean SSIM), equal to (F.l1_loss(img1, img2), fused_ssim(img1, img2, padding))
    up to floating-point summation order, from one forward and one backward kernel.

    CUDA/HIP, 2D, float32 only. Only img1 is differentiable: img2 must be a fixed target."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    assert padding in allowed_padding
    if not img1.is_cuda:
        raise ValueError("fused_l1_ssim requires CUDA/HIP inputs")
    if img1.device != img2.device or img1.dtype != torch.float32 or img2.dtype != torch.float32:
        raise ValueError("fused_l1_ssim requires float32 inputs on the same CUDA/HIP device")
    if img1.ndim != 4 or img1.shape != img2.shape:
        raise ValueError("fused_l1_ssim requires matching 4D image shapes")
    if img2.requires_grad:
        raise ValueError("fused_l1_ssim requires a fixed target (img2.requires_grad=False)")
    if not img1.is_contiguous(memory_format=torch.channels_last):
        img1 = img1.contiguous()
    return FusedL1SSIMMean.apply(C1, C2, img1, img2, 5 if padding == "valid" else 0)
