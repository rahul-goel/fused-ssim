#ifndef FusedSSIMSycl_HPP
#define FusedSSIMSycl_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

void
fusedssim_forward_kernel_call(
    int B, 
    int CH, 
    int H,
    int W,
    float C1,
    float C2,
    nb::ndarray<float> &img1,
    nb::ndarray<float> &img2,
    bool train,
    nb::ndarray<float> &ssim_map,
    nb::ndarray<float> &dm_dmu1,
    nb::ndarray<float> &dm_dsigma1_sq,
    nb::ndarray<float> &dm_dsigma12
);

void
fusedssim_backward_kernel_call(
    int B,
    int CH,
    int H,
    int W,
    float C1,
    float C2,
    nb::ndarray<float> &img1,
    nb::ndarray<float> &img2,
    nb::ndarray<float> &dL_dmap,
    nb::ndarray<float> &dm_dmu1,
    nb::ndarray<float> &dm_dsigma1_sq,
    nb::ndarray<float> &dm_dsigma12,
    nb::ndarray<float> &dL_dimg1
);

#endif //FusedSSIMSycl_HPP