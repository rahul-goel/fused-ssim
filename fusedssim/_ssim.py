from typing import NamedTuple
import torch.nn as nn
import torch

from . import _C

allowed_padding = ["same", "valid"]

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        # Call the C++ function
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = _C.fusedssim(C1, C2, img1, img2, train)

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
            new_dL_dmap = torch.zeros_like(img1)
            new_dL_dmap[:, :, 5:-5, 5:-5] = opt_grad
            dL_dmap = new_dL_dmap

        # Call the C++ function
        grad = _C.fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
        return None, None, grad, None, None, None, None # Match number of forward inputs

def fusedssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    assert padding in allowed_padding

    # Ensure tensors are contiguous in memory, which is often required by C++ extensions
    img1 = img1.contiguous()
    img2 = img2.contiguous()

    map_val = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map_val.mean()