import torch
import functools
import os
if os.name == 'nt':
    import sysconfig
    _dll1 = os.add_dll_directory(torch.__path__[0] + '/lib')
    # For sycl runtime
    _dll2 = os.add_dll_directory(sysconfig.get_paths()['platstdlib'] + '\\..\\Library\\bin')
from ._kernels import fusedssim_sycl_kernels as _sycl_kernels
if os.name == 'nt':
    _dll1.close()
    _dll2.close()

def detach_if_parameter(variable):
  if isinstance(variable, torch.nn.parameter.Parameter):
    return variable.detach()
  else:
    return variable


def detach_parameters_decorator(func):
  
  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    processed_args = [detach_if_parameter(arg) for arg in args]
    processed_kwargs = {key: detach_if_parameter(value) for key, value in kwargs.items()}
    return func(*processed_args, **processed_kwargs)
  
  return wrapper

@detach_parameters_decorator
def fusedssim_forward(C1, C2, img1, img2, train):
    B  = img1.shape[0]
    CH = img1.shape[1]
    H  = img1.shape[2]
    W  = img1.shape[3]

    ssim_map = torch.zeros_like(img1).contiguous()

    dm_dmu1 = torch.zeros_like(img1) if train  else torch.empty([0] )
    dm_dsigma1_sq = torch.zeros_like(img1) if train  else torch.empty([0])
    dm_dsigma12 = torch.zeros_like(img1) if train  else torch.empty([0])

    img1 = img1.contiguous()
    img2 = img2.contiguous()
    
    _sycl_kernels.fusedssim_forward_kernel_call(B,CH,H,W,C1,C2,img1, img2,train,ssim_map,dm_dmu1,dm_dsigma1_sq,dm_dsigma12)
    
    return ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12

@detach_parameters_decorator
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

    _sycl_kernels.fusedssim_backward_kernel_call(B,CH,H,W,C1,C2,img1, img2, dL_dmap, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, dL_dimg1)

    return dL_dimg1
