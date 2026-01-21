from typing import NamedTuple
import torch.nn as nn
import torch

if torch.cuda.is_available():
    from fused_ssim_cuda import fusedssim, fusedssim_backward, fusedssim3d, fusedssim_backward3d
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

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    img1 = img1.contiguous()
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train, 2)
    return map.mean()

