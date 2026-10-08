#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
);

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_sum(float C1, float C2, torch::Tensor &img1, torch::Tensor &img2, int pad, bool with_l1);

torch::Tensor
fusedssim_backward_mean(
    float C1, float C2, torch::Tensor &img1, torch::Tensor &img2, torch::Tensor &dL_dmean, double count, int pad,
    torch::Tensor &dm_dmu1, torch::Tensor &dm_dsigma1_sq, torch::Tensor &dm_dsigma12,
    c10::optional<torch::Tensor> dL_dl1, double l1_count);
