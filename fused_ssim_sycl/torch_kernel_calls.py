import torch

import sys 
sys.path.append("/home/adityasi/fused-ssim/build/fused_ssim_sycl_kernels")
from fused_ssim_sycl_kernels import fusedssim_forward_kernel_call, fusedssim_backward_kernel_call

def fusedssim_forward(C1, C2, img1, img2, train):
    B  = img1.shape[0]
    CH = img1.shape[1]
    H  = img1.shape[2]
    W  = img1.shape[3]

    ssim_map = torch.zeros_like(img1).contiguous()

    dm_dmu1 = torch.zeros_like(img1) if train  else torch.empty([0] )
    dm_dsigma1_sq = torch.zeros_like(img1) if train  else torch.empty([0])
    dm_dsigma12 = torch.zeros_like(img1) if train  else torch.empty([0])

    img1 = img1.detach().contiguous()
    img2 = img2.detach().contiguous()
    
    fusedssim_forward_kernel_call(B,CH,H,W,C1,C2,img1, img2,train,ssim_map,dm_dmu1,dm_dsigma1_sq,dm_dsigma12)
    
    return ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12

def fusedssim_backward(C1, C2, img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12):
    B  = img1.shape[0]
    CH = img1.shape[1]
    H  = img1.shape[2]
    W  = img1.shape[3] 

    dL_dimg1 = torch.zeros_like(img1)

    img1 = img1.contiguous()
    img2 = img2.contiguous()

    dL_dmap = dL_dmap.contiguous()
    dm_dmu1 = dm_dmu1.contiguous()
    dm_dsigma1_sq = dm_dsigma1_sq.contiguous()
    dm_dsigma12 = dm_dsigma12.contiguous()

    fusedssim_backward_kernel_call(B,CH,H,W,C1,C2,img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, dL_dimg1)

    return dL_dimg1