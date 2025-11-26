from typing import NamedTuple
import torch.nn as nn
import torch

from fused_ssim3d_cuda import fusedssim3d, fusedssim_backward3d, fusedssim2d, fusedssim_backward2d, fusedssim3d_by_2d, fusedssim_backward3d_by_2d


allowed_padding = ["same", "valid"]

    
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True, spatial_dims=2, mode="default"):
        if spatial_dims == 2:
            ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim2d(C1, C2, img1, img2, train)
        elif spatial_dims == 3:
            if mode == "default":
                ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim3d(C1, C2, img1, img2, train)
            elif mode == "by_2d":
                ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim3d_by_2d(C1, C2, img1, img2, train)
        
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
        ctx.mode = mode

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
            grad = fusedssim_backward2d(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        elif ctx.spatial_dims == 3:
            if ctx.mode == "default":
                grad = fusedssim_backward3d(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
            elif ctx.mode == "by_2d":
                grad = fusedssim_backward3d_by_2d(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None, None, None

def fused_ssim3d(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 3, "default")
    return map.mean()

def fused_ssim2d(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 2, "default")
    return map.mean()

def fused_ssim3d_by_2d(img1, img2, padding="same", train=True): 
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 3, "by_2d")
    return map.mean()
