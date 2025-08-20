#ifndef FusedSSIMSycl_BackwardKernel_HPP
#define FusedSSIMSycl_BackwardKernel_HPP

#include "utils.hpp"

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

#endif //FusedSSIMSycl_BackwardKernel_HPP