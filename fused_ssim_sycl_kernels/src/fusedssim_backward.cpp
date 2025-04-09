#include "bindings.hpp"
#include "kernels/FusedSSIMBackwardKernel.hpp"

#include <c10/xpu/XPUStream.h>
#include <torch/torch.h>
#include <torch/extension.h>

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
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
) {
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    auto dL_dimg1 = torch::zeros_like(img1);

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
            sycl::accessor<float, 3, sycl::access::mode::read_write, sycl::target::local> sData(sData_range, cgh); 

            sycl::range<3> sScratch_range(CONV_Y, CONV_X, 3);
            sycl::accessor<float, 3, sycl::access::mode::read_write, sycl::target::local> sScratch(sScratch_range, cgh);

            FusedSSIMBackwardKernel 
            kernel
            (
                H, W, CH, C1, C2,
                img1.contiguous().data_ptr<float>(),
                img2.contiguous().data_ptr<float>(),
                dL_dmap.contiguous().data_ptr<float>(),
                dL_dimg1.data_ptr<float>(),
                dm_dmu1.contiguous().data_ptr<float>(),
                dm_dsigma1_sq.contiguous().data_ptr<float>(),
                dm_dsigma12.contiguous().data_ptr<float>(),
                sData,
                sScratch
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();

    return dL_dimg1;
}