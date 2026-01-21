#include <torch/extension.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

namespace cg = cooperative_groups;

// ------------------------------------------
// Constant Memory for Gaussian Coefficients
// ------------------------------------------
__constant__ float cGauss[11] = {
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

#define WINDOW_LENGTH (2 * HALO + 1)

// ------------------------------------------
// Utility: Safe pixel fetch w/ zero padding
// ------------------------------------------
__device__ __forceinline__ float get_pix_value(
    const float* img, 
    int b, int c, int z, int y, int x,
    int CH, int H, int W, int D
) {
    if (x < 0 || x >= W || y < 0 || y >= H || z < 0 || z >= D) {
        return 0.0f;
    }
    return img[b * CH * H * W * D + c * H * W * D + z * H * W + y * W + x]; //ZYX ordering
}

__device__ __forceinline__ void load_stat_window(
    float* __restrict__ buf0,
    float* __restrict__ buf1,
    float* __restrict__ buf2,
    float* __restrict__ buf3,
    float* __restrict__ buf4,
    int slot,
    int abs_z,
    bool inside,
    const float* __restrict__ stat0,
    const float* __restrict__ stat1,
    const float* __restrict__ stat2,
    const float* __restrict__ stat3,
    const float* __restrict__ stat4,
    size_t channel_base,
    size_t plane_stride,
    size_t pix_offset,
    int D
) {
    if (!inside || abs_z < 0 || abs_z >= D) {
        buf0[slot] = 0.f;
        buf1[slot] = 0.f;
        buf2[slot] = 0.f;
        buf3[slot] = 0.f;
        buf4[slot] = 0.f;
        return;
    }
    const size_t idx = channel_base + static_cast<size_t>(abs_z) * plane_stride + pix_offset;
    buf0[slot] = stat0[idx];
    buf1[slot] = stat1[idx];
    buf2[slot] = stat2[idx];
    buf3[slot] = stat3[idx];
    buf4[slot] = stat4[idx];
}


__device__ __forceinline__ void load_back_window(
    float* __restrict__ buf0,
    float* __restrict__ buf1,
    float* __restrict__ buf2,
    int slot,
    int abs_z,
    bool inside,
    const float* __restrict__ stat0,
    const float* __restrict__ stat1,
    const float* __restrict__ stat2,
    size_t channel_base,
    size_t plane_stride,
    size_t pix_offset,
    int D
) {
    if (!inside || abs_z < 0 || abs_z >= D) {
        buf0[slot] = 0.f;
        buf1[slot] = 0.f;
        buf2[slot] = 0.f;
        return;
    }
    const size_t idx = channel_base + static_cast<size_t>(abs_z) * plane_stride + pix_offset;
    buf0[slot] = stat0[idx];
    buf1[slot] = stat1[idx];
    buf2[slot] = stat2[idx];
}



//    sigma1_sq, sigma2_sq, sigma12, etc.
//  - Writes final SSIM map to ssim_map
//  - Optionally writes partial derivatives
//    to dm_dmu1, dm_dsigma1_sq, dm_dsigma12
// ------------------------------------------
__global__ void fusedssim3dCUDA(
    int H, //height
    int W, //width
    int D, //depth
    int B, //batch size
    int CH, //channels
    float C1, //const for SSIM
    float C2, //const for SSIM
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    float* __restrict__ ssim_map,
    float* __restrict__ dm_dmu1,
    float* __restrict__ dm_dsigma1_sq,
    float* __restrict__ dm_dsigma12,
    float* __restrict__ xy_stats
) {
    auto block = cg::this_thread_block();
    // const int bIdx   = block.group_index().z;  // batch index
    const int bIdx = block.group_index().z;
    if (bIdx >= B) {
        return;
    }
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const bool inside = (pix_x < W) && (pix_y < H);

    const size_t plane_stride   = static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t channel_stride = static_cast<size_t>(D) * plane_stride;
    const size_t voxels_per_stat = static_cast<size_t>(B) * static_cast<size_t>(CH) * channel_stride;

    float* stat0 = xy_stats;
    float* stat1 = xy_stats + voxels_per_stat;
    float* stat2 = xy_stats + 2 * voxels_per_stat;
    float* stat3 = xy_stats + 3 * voxels_per_stat;
    float* stat4 = xy_stats + 4 * voxels_per_stat;

    const size_t bc_base   = (static_cast<size_t>(bIdx) * static_cast<size_t>(CH)) * channel_stride;
    const size_t pix_offset = inside ? (static_cast<size_t>(pix_y) * static_cast<size_t>(W) + pix_x) : size_t(0);

    // Shared memory for the tile (img1, img2)
    __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // After horizontal pass, store partial sums here
    // conv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    __shared__ float conv[CONV_Y][CONV_X][5];

    for (int c = 0; c < CH; ++c) {
        const size_t channel_base = bc_base + static_cast<size_t>(c) * channel_stride;
        for (int z = 0; z < D; ++z) {
            // ------------------------------------------------------------
            // 1) Load (img1, img2) tile + halo into shared memory
            // ------------------------------------------------------------
            {
                const int tileSize = SHARED_Y * SHARED_X;
                const int threads = BLOCK_X * BLOCK_Y;
                const int steps = (tileSize + threads - 1) / threads;

                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank(); //thread id
                    if (tid < tileSize) {
                        int local_y = tid / SHARED_X;
                        int local_x = tid % SHARED_X;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, bIdx, c, z, gy, gx, CH, H, W, D);
                        float Y = get_pix_value(img2, bIdx, c, z, gy, gx, CH, H, W, D);

                        sTile[local_y][local_x][0] = X;
                        sTile[local_y][local_x][1] = Y;
                    }
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 2) X axis convolution (11x1) in shared memory
            //    We'll accumulate symmetrical pairs around center.
            // ------------------------------------------------------------
             
            
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;  // skip left halo

            float sumX   = 0.f;
            float sumX2  = 0.f;
            float sumY   = 0.f;
            float sumY2  = 0.f;
            float sumXY  = 0.f;

            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Xright = sTile[ly][lx + d][0];
                float Yright = sTile[ly][lx + d][1];

                sumX  += (Xleft + Xright) * w;
                sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
            }
            // center
            {
                float centerX = sTile[ly][lx][0];
                float centerY = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX  += centerX * wc;
                sumX2 += (centerX * centerX) * wc;
                sumY  += centerY * wc;
                sumY2 += (centerY * centerY) * wc;
                sumXY += (centerX * centerY) * wc;
            }

            // Write out partial sums
            conv[ly][threadIdx.x][0] = sumX;
            conv[ly][threadIdx.x][1] = sumX2;
            conv[ly][threadIdx.x][2] = sumY;
            conv[ly][threadIdx.x][3] = sumY2;
            conv[ly][threadIdx.x][4] = sumXY;

            // Possibly handle second row in same warp
            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sumX   = 0.f; sumX2  = 0.f;
                sumY   = 0.f; sumY2  = 0.f;
                sumXY  = 0.f;

#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float Xleft  = sTile[ly2][lx - d][0];
                    float Yleft  = sTile[ly2][lx - d][1];
                    float Xright = sTile[ly2][lx + d][0];
                    float Yright = sTile[ly2][lx + d][1];

                    sumX  += (Xleft + Xright) * w;
                    sumX2 += ((Xleft * Xleft) + (Xright * Xright)) * w;
                    sumY  += (Yleft + Yright) * w;
                    sumY2 += ((Yleft * Yleft) + (Yright * Yright)) * w;
                    sumXY += ((Xleft * Yleft) + (Xright * Yright)) * w;
                }
                // center
                
                float cx = sTile[ly2][lx][0];
                float cy = sTile[ly2][lx][1];
                float wc = cGauss[HALO];
                sumX  += cx * wc;
                sumX2 += (cx * cx) * wc;
                sumY  += cy * wc;
                sumY2 += (cy * cy) * wc;
                sumXY += (cx * cy) * wc;
                
                conv[ly2][threadIdx.x][0] = sumX;
                conv[ly2][threadIdx.x][1] = sumX2;
                conv[ly2][threadIdx.x][2] = sumY;
                conv[ly2][threadIdx.x][3] = sumY2;
                conv[ly2][threadIdx.x][4] = sumXY;
            }
            
            block.sync();
            // ------------------------------------------------------------
            // 3) Y axis convolution (1x11)
            // ------------------------------------------------------------   
            
            ly = threadIdx.y + HALO;
            lx = threadIdx.x; 
            
            sumX = sumX2 = sumY = sumY2 = sumXY = 0.f;
            
            // #pragma unroll for those 5 pairs
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = conv[ly - d][lx];
                float* bot = conv[ly + d][lx];

                sumX += (top[0] + bot[0]) * w;
                sumX2 += (top[1] + bot[1]) * w;
                sumY += (top[2] + bot[2]) * w;
                sumY2 += (top[3] + bot[3]) * w;
                sumXY += (top[4] + bot[4]) * w;
            }

            float wC = cGauss[HALO];
            float* ctr = conv[ly][lx];
            sumX += ctr[0] * wC;
            sumX2 += ctr[1] * wC;
            sumY += ctr[2] * wC;
            sumY2 += ctr[3] * wC;
            sumXY += ctr[4] * wC;

            if (inside) {
                const size_t idx = channel_base + static_cast<size_t>(z) * plane_stride + pix_offset;
                stat0[idx] = sumX;
                stat1[idx] = sumX2;
                stat2[idx] = sumY;
                stat3[idx] = sumY2;
                stat4[idx] = sumXY;
            }
            
            block.sync();
        }    
        // ------------------------------------------------------------
        // 3) Z axis convolution (1x11) + final SSIM
        //    Performed using a ring buffer window in registers
        // ------------------------------------------------------------

        if (inside) {
            // Keep a sliding 11-slice (WINDOW_LENGTH) window in registers for each statistic.
            // `head` always points at the slot that represents the current absolute z plane in window,
            // while newer/older planes wrap around the buffer via modulo arithmetic.
            float ring0[WINDOW_LENGTH];
            float ring1[WINDOW_LENGTH];
            float ring2[WINDOW_LENGTH];
            float ring3[WINDOW_LENGTH];
            float ring4[WINDOW_LENGTH];
            
            int head = 0;
            // Prime the buffers with values for z in [-HALO, HALO] so every iteration can
            // assume the window is already centered on the current plane.
            for (int offset = -HALO; offset <= HALO; ++offset) {
                int slot = (head + offset + HALO) % WINDOW_LENGTH;
                load_stat_window(
                    ring0, ring1, ring2, ring3, ring4,
                    slot,
                    offset,
                    inside,
                    stat0, stat1, stat2, stat3, stat4,
                    channel_base,
                    plane_stride,
                    pix_offset,
                    D
                );
            }

            for (int z = 0; z < D; ++z) {
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
                float sum3 = 0.f;
                float sum4 = 0.f;
#pragma unroll
                for (int d = -HALO; d <= HALO; ++d) {
                    // Convert relative offset d into a ring slot: +HALO prevents negatives and
                    // modulo WINDOW_LENGTH wraps us around the 11-element circular window.
                    int slot = (head + d + HALO) % WINDOW_LENGTH;
                    int ad = (d < 0) ? -d : d;
                    float w = cGauss[HALO - ad];
                    sum0 += ring0[slot] * w;
                    sum1 += ring1[slot] * w;
                    sum2 += ring2[slot] * w;
                    sum3 += ring3[slot] * w;
                    sum4 += ring4[slot] * w;
                }

                const size_t idx = channel_base + static_cast<size_t>(z) * plane_stride + pix_offset;
                float mu1 = sum0;
                float mu2 = sum2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = sum1 - mu1_sq;
                float sigma2_sq = sum3 - mu2_sq;
                float sigma12   = sum4 - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);
                ssim_map[idx] = val;

                if (dm_dmu1) {
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);
                    dm_dmu1[idx]       = d_m_dmu1;
                    dm_dsigma1_sq[idx] = d_m_dsigma1_sq;
                    dm_dsigma12[idx]   = d_m_dsigma12;
                }

                // Slide the window forward by one plane: advance `head` (new center),
                // overwrite the slot that just wrapped around with the newly visible z+HALO+1.
                head = (head + 1) % WINDOW_LENGTH;
                int next_z = z + HALO + 1;
                int slot_new = (head + WINDOW_LENGTH - 1) % WINDOW_LENGTH;
                load_stat_window(
                    ring0, ring1, ring2, ring3, ring4,
                    slot_new,
                    next_z,
                    inside,
                    stat0, stat1, stat2, stat3, stat4,
                    channel_base,
                    plane_stride,
                    pix_offset,
                    D
                );
            }
        block.sync();
        }
    }
}

__global__ void fusedssim_backward3dCUDA(
    int H,
    int W,
    int D,
    int B,
    int CH,
    float C1,
    float C2,
    const float* __restrict__ img1,
    const float* __restrict__ img2,
    const float* __restrict__ dL_dmap,
    float* __restrict__ dL_dimg1,
    const float* __restrict__ dm_dmu1,
    const float* __restrict__ dm_dsigma1_sq,
    const float* __restrict__ dm_dsigma12,
    float* __restrict__ yz_stats
) {
    auto block = cg::this_thread_block();
    (void)C1;
    (void)C2;
    const int bIdx = block.group_index().z;
    if (bIdx >= B) {
        return;
    }

    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const bool inside = (pix_x < W) && (pix_y < H);

    const size_t plane_stride   = static_cast<size_t>(H) * static_cast<size_t>(W);
    const size_t channel_stride = static_cast<size_t>(D) * plane_stride;
    const size_t voxels_per_stat = static_cast<size_t>(B) * static_cast<size_t>(CH) * channel_stride;

    float* stat0 = yz_stats;
    float* stat1 = yz_stats + voxels_per_stat;
    float* stat2 = yz_stats + 2 * voxels_per_stat;

    const size_t bc_base   = (static_cast<size_t>(bIdx) * static_cast<size_t>(CH)) * channel_stride;
    const size_t pix_offset = inside ? (static_cast<size_t>(pix_y) * static_cast<size_t>(W) + pix_x) : size_t(0);

    __shared__ float sTile[SHARED_Y][SHARED_X][3];
    __shared__ float conv[CONV_Y][CONV_X][3];

    const int tileSize = SHARED_Y * SHARED_X;
    const int threads = BLOCK_X * BLOCK_Y;
    const int steps = (tileSize + threads - 1) / threads;
    const int tileStartY = block.group_index().y * BLOCK_Y;
    const int tileStartX = block.group_index().x * BLOCK_X;

    for (int c = 0; c < CH; ++c) {
        const size_t channel_base = bc_base + static_cast<size_t>(c) * channel_stride;

        for (int z = 0; z < D; ++z) {
            for (int s = 0; s < steps; ++s) {
                int tid = s * threads + block.thread_rank();
                if (tid < tileSize) {
                    int local_y = tid / SHARED_X;
                    int local_x = tid % SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;

                    float chain = get_pix_value(dL_dmap, bIdx, c, z, gy, gx, CH, H, W, D);
                    float vmu   = get_pix_value(dm_dmu1, bIdx, c, z, gy, gx, CH, H, W, D);
                    float vs1   = get_pix_value(dm_dsigma1_sq, bIdx, c, z, gy, gx, CH, H, W, D);
                    float vs12  = get_pix_value(dm_dsigma12, bIdx, c, z, gy, gx, CH, H, W, D);

                    sTile[local_y][local_x][0] = vmu  * chain;
                    sTile[local_y][local_x][1] = vs1  * chain;
                    sTile[local_y][local_x][2] = vs12 * chain;
                }
            }
            block.sync();

            //------------------------------------------------------------
            // (2) X axis pass
            //------------------------------------------------------------

            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;

            float sum0 = 0.f;
            float sum1 = 0.f;
            float sum2 = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float left0  = sTile[ly][lx - d][0];
                float left1  = sTile[ly][lx - d][1];
                float left2  = sTile[ly][lx - d][2];
                float right0 = sTile[ly][lx + d][0];
                float right1 = sTile[ly][lx + d][1];
                float right2 = sTile[ly][lx + d][2];

                sum0 += (left0 + right0) * w;
                sum1 += (left1 + right1) * w;
                sum2 += (left2 + right2) * w;
            }
            {
                float wc = cGauss[HALO];
                float c0 = sTile[ly][lx][0];
                float c1 = sTile[ly][lx][1];
                float c2 = sTile[ly][lx][2];
                sum0 += c0 * wc;
                sum1 += c1 * wc;
                sum2 += c2 * wc;
            }

            conv[ly][threadIdx.x][0] = sum0;
            conv[ly][threadIdx.x][1] = sum1;
            conv[ly][threadIdx.x][2] = sum2;

            int ly2 = ly + BLOCK_Y;
            if (ly2 < CONV_Y) {
                sum0 = sum1 = sum2 = 0.f;
#pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float left0  = sTile[ly2][lx - d][0];
                    float left1  = sTile[ly2][lx - d][1];
                    float left2  = sTile[ly2][lx - d][2];
                    float right0 = sTile[ly2][lx + d][0];
                    float right1 = sTile[ly2][lx + d][1];
                    float right2 = sTile[ly2][lx + d][2];

                    sum0 += (left0 + right0) * w;
                    sum1 += (left1 + right1) * w;
                    sum2 += (left2 + right2) * w;
                }
                float wc = cGauss[HALO];
                float c0 = sTile[ly2][lx][0];
                float c1 = sTile[ly2][lx][1];
                float c2 = sTile[ly2][lx][2];
                sum0 += c0 * wc;
                sum1 += c1 * wc;
                sum2 += c2 * wc;

                conv[ly2][threadIdx.x][0] = sum0;
                conv[ly2][threadIdx.x][1] = sum1;
                conv[ly2][threadIdx.x][2] = sum2;
            }
            block.sync();

            //----------------------------------------------------------
            // (2) Y axis pass
            //----------------------------------------------------------

            ly = threadIdx.y + HALO;
            lx = threadIdx.x;

            float out0 = 0.f;
            float out1 = 0.f;
            float out2 = 0.f;
#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = conv[ly - d][lx];
                float* bot = conv[ly + d][lx];

                out0 += (top[0] + bot[0]) * w;
                out1 += (top[1] + bot[1]) * w;
                out2 += (top[2] + bot[2]) * w;
            }
            {
                float wc = cGauss[HALO];
                float* ctr = conv[ly][lx];
                out0 += ctr[0] * wc;
                out1 += ctr[1] * wc;
                out2 += ctr[2] * wc;
            }

            if (inside) {
                const size_t idx = channel_base + static_cast<size_t>(z) * plane_stride + pix_offset;
                stat0[idx] = out0;
                stat1[idx] = out1;
                stat2[idx] = out2;
            }
            block.sync();
        }

        //----------------------------------------------------------
        // (3) Z axis pass -> finalize dL/d(img1)
        //----------------------------------------------------------

        if (inside) {
            // Same ring-buffer pattern as forward pass: each slot represents one z-slice of the
            // yz-filtered statistics, and `head` points at the current absolute z location.
            float ring0[WINDOW_LENGTH];
            float ring1[WINDOW_LENGTH];
            float ring2[WINDOW_LENGTH];
            int head = 0;
            // Preload the buffer so the loop can consume a fully populated window.
            for (int offset = -HALO; offset <= HALO; ++offset) {
                int slot = (head + offset + HALO) % WINDOW_LENGTH;
                load_back_window(
                    ring0, ring1, ring2,
                    slot,
                    offset,
                    inside,
                    stat0, stat1, stat2,
                    channel_base,
                    plane_stride,
                    pix_offset,
                    D
                );
            }

            for (int z = 0; z < D; ++z) {
                float sum0 = 0.f;
                float sum1 = 0.f;
                float sum2 = 0.f;
#pragma unroll
                for (int d = -HALO; d <= HALO; ++d) {
                    // Map the relative offset d back to the physical slot that cached it.
                    int slot = (head + d + HALO) % WINDOW_LENGTH;
                    int ad = (d < 0) ? -d : d;
                    float w = cGauss[HALO - ad];
                    sum0 += ring0[slot] * w;
                    sum1 += ring1[slot] * w;
                    sum2 += ring2[slot] * w;
                }

                float p1 = get_pix_value(img1, bIdx, c, z, pix_y, pix_x, CH, H, W, D);
                float p2 = get_pix_value(img2, bIdx, c, z, pix_y, pix_x, CH, H, W, D);
                const size_t idx = channel_base + static_cast<size_t>(z) * plane_stride + pix_offset;
                float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;
                dL_dimg1[idx] = dL_dpix;

                // Rotate the buffers: advance head and load the next plane entering the window.
                head = (head + 1) % WINDOW_LENGTH;
                int next_z = z + HALO + 1;
                int slot_new = (head + WINDOW_LENGTH - 1) % WINDOW_LENGTH;
                load_back_window(
                    ring0, ring1, ring2,
                    slot_new,
                    next_z,
                    inside,
                    stat0, stat1, stat2,
                    channel_base,
                    plane_stride,
                    pix_offset,
                    D
                );
            }
        }
        block.sync();
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim3d(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int D  = img1.size(2);
    int H  = img1.size(3);
    int W  = img1.size(4);

    // Launch config: each block covers one XY tile and marches across depth
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    // Output SSIM map
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    auto xy_workspace = torch::empty({5, B, CH, D, H, W}, img1.options()).contiguous();

    fusedssim3dCUDA<<<grid, block>>>(
        H, W, D, B, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr,
        xy_workspace.data_ptr<float>()
    );

    return std::make_tuple(ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// PyTorch Interface (Backward)
//   Takes the gradient wrt the SSIM map and
//   the partial derivatives from forward;
//   returns dL/d(img1).
// ------------------------------------------
torch::Tensor
fusedssim_backward3d(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    torch::Tensor &dL_dmap,
    torch::Tensor &dm_dmu1,
    torch::Tensor &dm_dsigma1_sq,
    torch::Tensor &dm_dsigma12
) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(img1));
    int B  = img1.size(0);
    int CH = img1.size(1);
    int D  = img1.size(2);
    int H  = img1.size(3);
    int W  = img1.size(4);

    auto dL_dimg1 = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              B);
    dim3 block(BLOCK_X, BLOCK_Y);

    auto back_workspace = torch::empty({3, B, CH, D, H, W}, img1.options()).contiguous();

    fusedssim_backward3dCUDA<<<grid, block>>>(
        H, W, D, B, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>(),
        back_workspace.data_ptr<float>()
    );

    return dL_dimg1;
}
