#ifndef FusedSSIMSycl_HPP
#define FusedSSIMSycl_HPP

#include <torch/extension.h>
#include <nanobind/nanobind.h>
#include <tuple>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim_forward_kernel_call(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

torch::Tensor
fusedssim_backward_kernel_call(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

#endif //FusedSSIMSycl_HPP