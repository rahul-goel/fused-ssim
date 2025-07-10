#include "bindings.hpp"
#include "kernels/FusedSSIMForwardKernel.hpp"

#include <c10/xpu/XPUStream.h>


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
) {

    sycl::range<3> localRange{
        BLOCK_X, 
        BLOCK_Y, 
        1
    };

    sycl::range<3> globalRange{ 
        static_cast<size_t>(((W + BLOCK_X - 1) / BLOCK_X)*BLOCK_X), 
        static_cast<size_t>(((H + BLOCK_Y - 1) / BLOCK_Y)*BLOCK_Y), 
        static_cast<size_t>(B)
    };

    sycl::nd_range<3> range(globalRange, localRange);

    auto d_queue = at::xpu::getCurrentXPUStream().queue();
    auto e = d_queue.submit(
        [&](sycl::handler& cgh)
        {
            sycl::range<3> sTile_range(SHARED_Y, SHARED_X, 2);
            sycl::local_accessor<float, 3> sTile(sTile_range, cgh);

            sycl::range<3> xconv_range(CONV_Y, CONV_X, 5);
            sycl::local_accessor<float, 3> xconv(xconv_range, cgh);
            
            FusedSSIMForwardKernel 
            kernel
            (
                H, W, CH, C1, C2,
                img1.data(),
                img2.data(),
                ssim_map.data(),
                train ? dm_dmu1.data()       : nullptr,
                train ? dm_dsigma1_sq.data() : nullptr,
                train ? dm_dsigma12.data()   : nullptr,
                sTile,
                xconv   
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();
}
