#include <torch/extension.h>
#include "ssim.h"

#include <sycl/sycl.hpp>
#include <c10/xpu/XPUStream.h>

// ------------------------------------------
// Block and Shared Memory Dimensions
// ------------------------------------------
#define BLOCK_X 16
#define BLOCK_Y 16
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y

// ------------------------------------------
// Gaussian Coefficients
// ------------------------------------------
constexpr float cGauss[11] = {
    0.001028380123898387f,
    0.0075987582094967365f,
    0.036000773310661316f,
    0.10936068743467331f,
    0.21300552785396576f,
    0.26601171493530273f,
    0.21300552785396576f,
    0.10936068743467331f,
    0.036000773310661316f,
    0.0075987582094967365f,
    0.001028380123898387f
};


// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
inline float get_pix_value(
    const float* img, 
    int b, int c, int y, int x,
    int CH, int H, int W
) {
    if (x < 0 || x >= W || y < 0 || y >= H) {
        return 0.0f;
    }
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// ------------------------------------------
// Forward Kernel: Fused SSIM
//  - Two-pass convolution to get mu1, mu2,
//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
struct FusedSSIMForwardKernel {

    int m_H;
    int m_W;
    int m_CH;
    float m_C1;
    float m_C2;
    const float* m_img1;
    const float* m_img2;
    float* m_ssim_map;
    float* m_dm_dmu1;
    float* m_dm_dsigma1_sq;
    float* m_dm_dsigma12;
    sycl::local_accessor<float, 3> m_sTile;
    sycl::local_accessor<float, 3> m_xconv;

    FusedSSIMForwardKernel (
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* img1,
        const float* img2,
        float* ssim_map,
        float* dm_dmu1,
        float* dm_dsigma1_sq,
        float* dm_dsigma12,
        sycl::local_accessor<float, 3> sTile,
        sycl::local_accessor<float, 3> xconv
    ) :
        m_H(H),
        m_W(W),
        m_CH(CH),
        m_C1(C1),
        m_C2(C2),
        m_img1(img1),
        m_img2(img2),
        m_ssim_map(ssim_map),
        m_dm_dmu1(dm_dmu1),
        m_dm_dsigma1_sq(dm_dsigma1_sq),
        m_dm_dsigma12(dm_dsigma12),
        m_sTile(sTile),
        m_xconv(xconv)
    {}

    void operator()(sycl::nd_item<3> work_item) const {
        const int bIdx   = work_item.get_group(2);  // batch index
        const int pix_y  = work_item.get_group(1) * BLOCK_Y + work_item.get_local_id(1);
        const int pix_x  = work_item.get_group(0) * BLOCK_X + work_item.get_local_id(0);
        const int pix_id = pix_y * m_W + pix_x;
        const int num_pix = m_H * m_W;

        for( int c = 0; c < m_CH; ++c)
        {
            
            // ------------------------------------------------------------
            // 1) Load (img1, img2) tile + halo into shared memory
            // ------------------------------------------------------------

            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;

                const int tileStartY = work_item.get_group(1) * BLOCK_Y;
                const int tileStartX = work_item.get_group(0) * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + work_item.get_local_linear_id();
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(m_img1, bIdx, c, gy, gx, m_CH, m_H, m_W);
                        float Y = get_pix_value(m_img2, bIdx, c, gy, gx, m_CH, m_H, m_W);

                        m_sTile[local_y][local_x][0] = X;
                        m_sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            work_item.barrier(sycl::access::fence_space::local_space);

            // ------------------------------------------------------------
            // 2) Horizontal convolution (11x1) in shared memory
            //    We'll accumulate symmetrical pairs around center.
            // ------------------------------------------------------------
            {
                int ly = work_item.get_local_id(1);
                int lx = work_item.get_local_id(0) + HALO;  // skip left halo

                float sumX   = 0.f;
                float sumX2  = 0.f;
                float sumY   = 0.f;
                float sumY2  = 0.f;
                float sumXY  = 0.f;

                #pragma unroll
                for(int d = 1; d <= HALO; ++d)
                {
                    float w = cGauss[HALO - d];
                    float Xleft  = m_sTile[ly][lx - d][0];
                    float Yleft  = m_sTile[ly][lx - d][1];
                    float Xright = m_sTile[ly][lx + d][0];
                    float Yright = m_sTile[ly][lx + d][1];

                    sumX  += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY  += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                {
                    float centerX = m_sTile[ly][lx][0];
                    float centerY = m_sTile[ly][lx][1];
                    float wc = cGauss[HALO];
                    sumX  += centerX * wc;
                    sumX2 += (centerX * centerX) * wc;
                    sumY  += centerY * wc;
                    sumY2 += (centerY * centerY) * wc;
                    sumXY += (centerX * centerY) * wc;
                }
                // Write out partial sums
                m_xconv[ly][work_item.get_local_id(0)][0] = sumX;
                m_xconv[ly][work_item.get_local_id(0)][1] = sumX2;
                m_xconv[ly][work_item.get_local_id(0)][2] = sumY;
                m_xconv[ly][work_item.get_local_id(0)][3] = sumY2;
                m_xconv[ly][work_item.get_local_id(0)][4] = sumXY;

                // Possibly handle second row in same warp
                int ly2 = ly + BLOCK_Y;
                if (ly2 < CONV_Y)
                {
                    sumX   = 0.f; sumX2  = 0.f;
                    sumY   = 0.f; sumY2  = 0.f;
                    sumXY  = 0.f;

                    #pragma unroll
                    for (int d = 1; d <= HALO; ++d) {
                        float w = cGauss[HALO - d];
                        float Xleft  = m_sTile[ly2][lx - d][0];
                        float Yleft  = m_sTile[ly2][lx - d][1];
                        float Xright = m_sTile[ly2][lx + d][0];
                        float Yright = m_sTile[ly2][lx + d][1];
    
                        sumX  += (Xleft + Xright) * w;
                        sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                        sumY  += (Yleft + Yright) * w;
                        sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                        sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                    }
                    // center
                    {
                        float cx = m_sTile[ly2][lx][0];
                        float cy = m_sTile[ly2][lx][1];
                        float wc = cGauss[HALO];
                        sumX  += cx * wc;
                        sumX2 += (cx * cx) * wc;
                        sumY  += cy * wc;
                        sumY2 += (cy * cy) * wc;
                        sumXY += (cx * cy) * wc;
                    }
                    m_xconv[ly2][work_item.get_local_id(0)][0] = sumX;
                    m_xconv[ly2][work_item.get_local_id(0)][1] = sumX2;
                    m_xconv[ly2][work_item.get_local_id(0)][2] = sumY;
                    m_xconv[ly2][work_item.get_local_id(0)][3] = sumY2;
                    m_xconv[ly2][work_item.get_local_id(0)][4] = sumXY;
                }
            }
            work_item.barrier(sycl::access::fence_space::local_space);

            // ------------------------------------------------------------
            // 3) Vertical convolution (1x11) + final SSIM
            // ------------------------------------------------------------
            {
                int ly = work_item.get_local_id(1) + HALO;
                int lx = work_item.get_local_id(0);

                float out0 = 0.f, out1 = 0.f, out2 = 0.f, out3 = 0.f, out4 = 0.f;
                
                #pragma unroll
                for(int d = 1; d <= HALO; ++d)
                {
                    float w = cGauss[HALO - d];

                    out0 += (m_xconv[ly - d][lx][0] + m_xconv[ly + d][lx][0]) * w;
                    out1 += (m_xconv[ly - d][lx][1] + m_xconv[ly + d][lx][1]) * w;
                    out2 += (m_xconv[ly - d][lx][2] + m_xconv[ly + d][lx][2]) * w;
                    out3 += (m_xconv[ly - d][lx][3] + m_xconv[ly + d][lx][3]) * w;
                    out4 += (m_xconv[ly - d][lx][4] + m_xconv[ly + d][lx][4]) * w;
                }
                //center 
                {
                    float wC = cGauss[HALO];
                    out0 += m_xconv[ly][lx][0] * wC;
                    out1 += m_xconv[ly][lx][1] * wC;
                    out2 += m_xconv[ly][lx][2] * wC;
                    out3 += m_xconv[ly][lx][3] * wC;
                    out4 += m_xconv[ly][lx][4] * wC;
                }

                if (pix_x < m_W && pix_y < m_H)
                {
                    float mu1 = out0;
                    float mu2 = out2;
                    float mu1_sq = mu1 * mu1;
                    float mu2_sq = mu2 * mu2;

                    float sigma1_sq = out1 - mu1_sq;
                    float sigma2_sq = out3 - mu2_sq;
                    float sigma12   = out4 - mu1 * mu2;

                    float A = mu1_sq + mu2_sq + m_C1;
                    float B = sigma1_sq + sigma2_sq + m_C2;
                    float C_ = 2.f * mu1 * mu2 + m_C1;
                    float D_ = 2.f * sigma12 + m_C2;

                    float val = (C_ * D_) / (A * B);

                    int global_idx = bIdx * m_CH * num_pix + c * num_pix + pix_id;
                    m_ssim_map[global_idx] = val;

                    if (m_dm_dmu1) {
                        // partial derivatives
                        float d_m_dmu1 = (
                            (mu2 * 2.f * D_) / (A * B)
                            - (mu2 * 2.f * C_) / (A * B)
                            - (mu1 * 2.f * C_ * D_) / (A * A * B)
                            + (mu1 * 2.f * C_ * D_) / (A * B * B)
                        );
                        float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                        float d_m_dsigma12   = (2.f * C_) / (A * B);
    
                        m_dm_dmu1[global_idx]       = d_m_dmu1;
                        m_dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                        m_dm_dsigma12[global_idx]   = d_m_dsigma12;
                    }
                }
            }
        }
    }
};

// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
//    and dL/dmap (the gradient from above).
// ------------------------------------------
struct FusedSSIMBackwardKernel{

    int m_H;
    int m_W;
    int m_CH;
    float m_C1;
    float m_C2;
    const float* m_img1;
    const float* m_img2;
    const float* m_dL_dmap;
    float* m_dL_dimg1;
    const float* m_dm_dmu1;
    const float* m_dm_dsigma1_sq;
    const float* m_dm_dsigma12;
    sycl::local_accessor<float, 3> m_sData;
    sycl::local_accessor<float, 3> m_sScratch;

    FusedSSIMBackwardKernel(
        int H,
        int W,
        int CH,
        float C1,
        float C2,
        const float* img1,
        const float* img2,
        const float* dL_dmap,
        float* dL_dimg1,
        const float* dm_dmu1,
        const float* dm_dsigma1_sq,
        const float* dm_dsigma12,
        sycl::local_accessor<float, 3> sData,
        sycl::local_accessor<float, 3> sScratch
    ) 
    : 
        m_H(H),
        m_W(W),
        m_CH(CH),
        m_C1(C1),
        m_C2(C2),
        m_img1(img1),
        m_img2(img2),
        m_dL_dmap(dL_dmap),
        m_dL_dimg1(dL_dimg1),
        m_dm_dmu1(dm_dmu1),
        m_dm_dsigma1_sq(dm_dsigma1_sq),
        m_dm_dsigma12(dm_dsigma12),
        m_sData(sData),
        m_sScratch(sScratch)
    {}

    void operator()(sycl::nd_item<3> work_item) const {
        const int bIdx   = work_item.get_group(2);  // batch index
        const int pix_y  = work_item.get_group(1) * BLOCK_Y + work_item.get_local_id(1);
        const int pix_x  = work_item.get_group(0) * BLOCK_X + work_item.get_local_id(0);
        const int pix_id = pix_y * m_W + pix_x;
        const int num_pix = m_H * m_W;

        for(int c = 0; c < m_CH; ++c)
        {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < m_W && pix_y < m_H) {
                p1 = get_pix_value(m_img1, bIdx, c, pix_y, pix_x, m_CH, m_H, m_W);
                p2 = get_pix_value(m_img2, bIdx, c, pix_y, pix_x, m_CH, m_H, m_W);
            }

            // (1) Load + fuse multiplication
            {
                const int start_y = work_item.get_group(1) * BLOCK_Y;
                const int start_x = work_item.get_group(0) * BLOCK_X;

                int tid = work_item.get_local_id(1) * work_item.get_local_range(0) + work_item.get_local_id(0);
                int warp_id = tid / 32;
                int lane_id = tid % 32;
                int totalThreads = BLOCK_X * BLOCK_Y;
                int num_warps = (totalThreads + 31) / 32;

                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;
    
                        float chain = get_pix_value(m_dL_dmap,      bIdx, c, gy, gx, m_CH, m_H, m_W);
                        float vmu   = get_pix_value(m_dm_dmu1,      bIdx, c, gy, gx, m_CH, m_H, m_W);
                        float vs1   = get_pix_value(m_dm_dsigma1_sq,bIdx, c, gy, gx, m_CH, m_H, m_W);
                        float vs12  = get_pix_value(m_dm_dsigma12,  bIdx, c, gy, gx, m_CH, m_H, m_W);
    
                        m_sData[0][row][col] = vmu  * chain;
                        m_sData[1][row][col] = vs1  * chain;
                        m_sData[2][row][col] = vs12 * chain;
                    }
                }
            }
            work_item.barrier(sycl::access::fence_space::local_space);

            // (2) Horizontal pass
            {
                int ly = work_item.get_local_id(1);
                int lx = work_item.get_local_id(0) + HALO;
                for( int pass = 0; pass < 2; ++pass)
                {
                    int yy = ly + pass * BLOCK_Y;
                    if (yy < CONV_Y)
                    {
                        float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;
                        #pragma unroll
                        for( int d = 1; d <= HALO; ++d )
                        {
                            float w = cGauss[HALO - d];
                            float left0  = m_sData[0][yy][lx - d];
                            float left1  = m_sData[1][yy][lx - d];
                            float left2  = m_sData[2][yy][lx - d];

                            float right0 = m_sData[0][yy][lx + d];
                            float right1 = m_sData[1][yy][lx + d];
                            float right2 = m_sData[2][yy][lx + d];

                            accum0 += (left0 + right0) * w;
                            accum1 += (left1 + right1) * w;
                            accum2 += (left2 + right2) * w;
                        }
                        //center
                        {
                            float wc = cGauss[HALO];
                            float c0 = m_sData[0][yy][lx];
                            float c1 = m_sData[1][yy][lx];
                            float c2 = m_sData[2][yy][lx];
                            accum0 += c0 * wc;
                            accum1 += c1 * wc;
                            accum2 += c2 * wc;
                        }

                        m_sScratch[yy][work_item.get_local_id(0)][0] = accum0;
                        m_sScratch[yy][work_item.get_local_id(0)][1] = accum1;
                        m_sScratch[yy][work_item.get_local_id(0)][2] = accum2;
                    }
                }
            }
            work_item.barrier(sycl::access::fence_space::local_space);

            // (3) Vertical pass -> finalize dL/d(img1)
            if (pix_x < m_W && pix_y < m_H) {
                int ly = work_item.get_local_id(1) + HALO;
                int lx = work_item.get_local_id(0);

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

                #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];

                    sum0 += (m_sScratch[ly - d][lx][0] + m_sScratch[ly + d][lx][0]) * w;
                    sum1 += (m_sScratch[ly - d][lx][1] + m_sScratch[ly + d][lx][1]) * w;
                    sum2 += (m_sScratch[ly - d][lx][2] + m_sScratch[ly + d][lx][2]) * w;
                }
                // center
                {
                    float wc = cGauss[HALO];
                    sum0 += m_sScratch[ly][lx][0] * wc;
                    sum1 += m_sScratch[ly][lx][1] * wc;
                    sum2 += m_sScratch[ly][lx][2] * wc;
                }

                // final accumulation
                float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

                int out_idx = bIdx * m_CH * num_pix + c * num_pix + pix_id;
                m_dL_dimg1[out_idx] = dL_dpix;
            }
            work_item.barrier(sycl::access::fence_space::local_space);
        }
    }
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    // Get dimensions
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    // Create output tensors (empty for now)
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    // Get data pointers to contiguous tensors
    float* img1_ptr = img1.contiguous().data_ptr<float>();
    float* img2_ptr = img2.contiguous().data_ptr<float>();

    // Declare kernel launch parameters
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

    // launch the kernel and wait for it to terminate
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
                img1_ptr,
                img2_ptr,
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

    // Get dimensions
    int B  = img1.size(0);
    int CH = img1.size(1);
    int H  = img1.size(2);
    int W  = img1.size(3);

    // Create output gradient tensor (empty for now)
    auto dL_dimg1 = torch::zeros_like(img1);

    // Get data pointers to contiguous tensors
    float* img1_ptr = img1.contiguous().data_ptr<float>();
    float* img2_ptr = img2.contiguous().data_ptr<float>();
    float* dL_dmap_ptr = dL_dmap.contiguous().data_ptr<float>();
    float* dm_dmu1_ptr = dm_dmu1.contiguous().data_ptr<float>();
    float* dm_dsigma1_sq_ptr = dm_dsigma1_sq.contiguous().data_ptr<float>();
    float* dm_dsigma12_ptr = dm_dsigma12.contiguous().data_ptr<float>();

    // Declare kernel launch parameters
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

    // launch the kernel and wait for it to terminate
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
                img1_ptr,
                img2_ptr,
                dL_dmap_ptr,
                dL_dimg1.data_ptr<float>(),
                dm_dmu1_ptr,
                dm_dsigma1_sq_ptr,
                dm_dsigma12_ptr,
                sData,
                sScratch
            );
            cgh.parallel_for(range, kernel);
        }
    );
    e.wait();

    return dL_dimg1;
}
