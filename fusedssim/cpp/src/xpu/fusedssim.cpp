#include <torch/extension.h>
#include <c10/xpu/XPUStream.h>

#include "fusedssim/xpu/FusedSSIMForwardKernel.hpp"

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

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
                img1.contiguous().data_ptr<float>(),
                img2.contiguous().data_ptr<float>(),
                ssim_map.data_ptr<float>(),
                train ? dm_dmu1.data_ptr<float>()       : nullptr,
                train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
                train ? dm_dsigma12.data_ptr<float>()   : nullptr,
                sTile,
                xconv   
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}