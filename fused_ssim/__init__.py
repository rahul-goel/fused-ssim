from typing import NamedTuple
import torch.nn as nn
import torch

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward
    from fused_ssim_cuda import fusedl1ssim_loss, fusedl1ssim_loss_backward
elif torch.mps.is_available():
    from fused_ssim_mps import fusedssim, fusedssim_backward
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    from fused_ssim_xpu import fusedssim, fusedssim_backward


allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)

        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]

        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding

        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        C1, C2, padding = ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()


class FusedL1SSIMLossMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ssim_weight, C1, C2, img1, img2, padding="same", train=True):
        if not torch.cuda.is_available():
            raise RuntimeError("FusedL1SSIMLossMap only supports CUDA")

        l1_ssim_loss_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedl1ssim_loss(ssim_weight, C1, C2, img1, img2, train)
        if padding == "valid":
            l1_ssim_loss_map = l1_ssim_loss_map[:, :, 5:-5, 5:-5]
        ctx.save_for_backward(img1.detach(), img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        ctx.ssim_weight = ssim_weight
        ctx.C1 = C1
        ctx.C2 = C2
        ctx.padding = padding
        return l1_ssim_loss_map

    @staticmethod
    def backward(ctx, opt_grad):
        if not torch.cuda.is_available():
            raise RuntimeError("FusedL1SSIMLossMap only supports CUDA")

        img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = ctx.saved_tensors
        ssim_weight, C1, C2, padding = ctx.ssim_weight, ctx.C1, ctx.C2, ctx.padding
        dL_dmap = opt_grad
        if padding == "valid":
            dL_dmap = torch.zeros_like(img1)
            dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
        grad = fusedl1ssim_loss_backward(ssim_weight, C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, None, grad, None, None, None

def fused_l1_ssim_loss(img1, img2, ssim_weight=0.2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    map = FusedL1SSIMLossMap.apply(ssim_weight, C1, C2, img1, img2, padding, train)
    return map.mean()
