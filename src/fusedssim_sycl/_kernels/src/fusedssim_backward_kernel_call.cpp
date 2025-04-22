#include "bindings.hpp"
#include "kernels/FusedSSIMBackwardKernel.hpp"

#include <c10/xpu/XPUStream.h>

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
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

    auto& d_queue = at::xpu::getCurrentXPUStream().queue();
    auto e = d_queue.submit(
        [&](sycl::handler& cgh)
        {
            sycl::range<3> sData_range(3, SHARED_Y, SHARED_X);
            sycl::local_accessor<float, 3> sData(sData_range, cgh); 

            sycl::range<3> sScratch_range(CONV_Y, CONV_X, 3);
            sycl::local_accessor<float, 3> sScratch(sScratch_range, cgh);

            FusedSSIMBackwardKernel 
            kernel
            (
                H, W, CH, C1, C2,
                img1.data(),
                img2.data(),
                dL_dmap.data(),
                dL_dimg1.data(),
                dm_dmu1.data(),
                dm_dsigma1_sq.data(),
                dm_dsigma12.data(),
                sData,
                sScratch
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();
}