#ifndef FusedSSIMSycl_ForwardKernel_HPP
#define FusedSSIMSycl_ForwardKernel_HPP

#include <sycl/sycl.hpp>
#include "fusedssim/xpu/utils.hpp"

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

#endif //FusedSSIMSycl_ForwardKernel_HPP