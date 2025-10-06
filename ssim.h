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

#pragma once
static char *MPS_KERNEL = R"FUSED_SSIM_MPS(
#include <metal_stdlib>
using namespace metal;

// Constants
constant int BLOCK_X = 16;
constant int BLOCK_Y = 16;
constant int HALO    = 5;
constant int SHARED_X = BLOCK_X + 2 * HALO;
constant int SHARED_Y = BLOCK_Y + 2 * HALO;
constant int CONV_X = BLOCK_X;
constant int CONV_Y = SHARED_Y;

// Gaussian coefficients
constant float cGauss[11] = {
    0.00102838f, 0.00759876f, 0.03600077f, 0.10936069f,
    0.21300553f, 0.26601172f, 0.21300553f, 0.10936069f,
    0.03600077f, 0.00759876f, 0.00102838f
};

// Safe fetch with zero padding
inline float get_pix_value(const device float* img,
                           int b, int c, int y, int x,
                           int CH, int H, int W)
{
    if (x < 0 || x >= W || y < 0 || y >= H) return 0.0f;
    return img[b * CH * H * W + c * H * W + y * W + x];
}

// Forward kernel
kernel void fusedssim_forward(
      device const float* img1           [[buffer(0)]],
      device const float* img2           [[buffer(1)]],
      constant float&     C1             [[buffer(2)]],
      constant float&     C2             [[buffer(3)]],
      constant int&       H              [[buffer(4)]],
      constant int&       W              [[buffer(5)]],
      constant int&       CH             [[buffer(6)]],
      constant int&       B              [[buffer(7)]],
      device float*       ssim_map       [[buffer(8)]],
      device float*       dm_dmu1        [[buffer(9)]],
      device float*       dm_dsigma1_sq  [[buffer(10)]],
      device float*       dm_dsigma12    [[buffer(11)]],
      uint3               gid            [[thread_position_in_grid]],
      uint3               tid            [[thread_position_in_threadgroup]],
      uint3               tgroup_size    [[threads_per_threadgroup]],
      uint3               tgroup_pos     [[threadgroup_position_in_grid]],
      uint                thread_index   [[thread_index_in_threadgroup]]
)
{

    const int pix_x = gid.x;
    const int pix_y = gid.y;
    const int b = gid.z;
    const int bIdx  = tgroup_pos.z;
    const int pix_id = pix_y * W + pix_x;
    const int num_pix = H * W;

    threadgroup float sTile[SHARED_Y][SHARED_X][2];
    threadgroup float xconv[CONV_Y][CONV_X][5];

    for (int c = 0; c < CH; ++c)
    {
        // ------------------------------------------------------------
        // 1) Load (img1, img2) tile + halo into shared memory
        // ------------------------------------------------------------
        {
            const int tileSize = SHARED_X * SHARED_Y;
            const int threads = tgroup_size.x * tgroup_size.y;
            const int steps = (tileSize + threads - 1) / threads;
            
            const int tileStartY = tgroup_pos.y * tgroup_size.y;
            const int tileStartX = tgroup_pos.x * tgroup_size.x;
            
            for (int s = 0; s < steps; ++s)
            {
                int tid_global = s * threads + thread_index;
                if (tid_global < tileSize)
                {
                    int local_y = tid_global / SHARED_X;
                    int local_x = tid_global % SHARED_X;
                    int gy = tileStartY + local_y - HALO;
                    int gx = tileStartX + local_x - HALO;

                    float X = get_pix_value(img1, bIdx, c, gy, gx, CH, H, W);
                    float Y = get_pix_value(img2, bIdx, c, gy, gx, CH, H, W);

                    sTile[local_y][local_x][0] = X;
                    sTile[local_y][local_x][1] = Y;
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------------------------------------------------------------
        // 2) Horizontal convolution (11x1) in shared memory
        //    We'll accumulate symmetrical pairs around center.
        // ------------------------------------------------------------
        {
            int ly = tid.y;
            int lx = tid.x + HALO;
            
            float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;
            
            for (int d = 1; d <= HALO; ++d)
            {
                float w = cGauss[HALO - d];
                float Xleft  = sTile[ly][lx - d][0];
                float Xright = sTile[ly][lx + d][0];
                float Yleft  = sTile[ly][lx - d][1];
                float Yright = sTile[ly][lx + d][1];

                sumX  += (Xleft + Xright) * w;
                sumX2 += (Xleft*Xleft + Xright*Xright) * w;
                sumY  += (Yleft + Yright) * w;
                sumY2 += (Yleft*Yleft + Yright*Yright) * w;
                sumXY += (Xleft*Yleft + Xright*Yright) * w;
            }
            
            // center
            {
                float cx = sTile[ly][lx][0];
                float cy = sTile[ly][lx][1];
                float wc = cGauss[HALO];
                sumX  += cx * wc;
                sumX2 += cx*cx * wc;
                sumY  += cy * wc;
                sumY2 += cy*cy * wc;
                sumXY += cx*cy * wc;
            }

            // Write out partial sums
            xconv[ly][tid.x][0] = sumX;
            xconv[ly][tid.x][1] = sumX2;
            xconv[ly][tid.x][2] = sumY;
            xconv[ly][tid.x][3] = sumY2;
            xconv[ly][tid.x][4] = sumXY;
            
            // Possibly handle second row in same warp
            int ly2 = ly + tgroup_size.y;
            if (ly2 < CONV_Y) {
                float sumX = 0.f, sumX2 = 0.f, sumY = 0.f, sumY2 = 0.f, sumXY = 0.f;

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
                {
                    float cx = sTile[ly2][lx][0];
                    float cy = sTile[ly2][lx][1];
                    float wc = cGauss[HALO];
                    sumX  += cx * wc;
                    sumX2 += (cx * cx) * wc;
                    sumY  += cy * wc;
                    sumY2 += (cy * cy) * wc;
                    sumXY += (cx * cy) * wc;
                }
                xconv[ly2][tid.x][0] = sumX;
                xconv[ly2][tid.x][1] = sumX2;
                xconv[ly2][tid.x][2] = sumY;
                xconv[ly2][tid.x][3] = sumY2;
                xconv[ly2][tid.x][4] = sumXY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ------------------------------------------------------------
        // 3) Vertical convolution (1x11) + final SSIM
        // ------------------------------------------------------------
        {
            int ly = tid.y + HALO;
            int lx = tid.x;
            
            float out0=0.f, out1=0.f, out2=0.f, out3=0.f, out4=0.f;
            
            for (int d=1; d<=HALO; ++d)
            {
                float w = cGauss[HALO - d];
                threadgroup float* top = xconv[ly - d][lx];
                threadgroup float* bot = xconv[ly + d][lx];

                out0 += (top[0]+bot[0])*w;
                out1 += (top[1]+bot[1])*w;
                out2 += (top[2]+bot[2])*w;
                out3 += (top[3]+bot[3])*w;
                out4 += (top[4]+bot[4])*w;
            }
            //center
            {
                float wC = cGauss[HALO];
                threadgroup float* ctr = xconv[ly][lx];
                out0 += ctr[0] * wC;
                out1 += ctr[1] * wC;
                out2 += ctr[2] * wC;
                out3 += ctr[3] * wC;
                out4 += ctr[4] * wC;
            }
            
            if (pix_x < W && pix_y < H)
            {
                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                
                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12   = out4 - mu1*mu2;
                
                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f*mu1*mu2 + C1;
                float D_ = 2.f*sigma12 + C2;
                
                float val = (C_ * D_) / (A * B);
                
                int global_idx = bIdx*CH*num_pix + c*num_pix + pix_id;
                ssim_map[global_idx] = val;
                
                if (dm_dmu1 != nullptr) {
                    // partial derivatives
                    float d_m_dmu1 = (
                        (mu2 * 2.f * D_) / (A * B)
                        - (mu2 * 2.f * C_) / (A * B)
                        - (mu1 * 2.f * C_ * D_) / (A * A * B)
                        + (mu1 * 2.f * C_ * D_) / (A * B * B)
                    );
                    float d_m_dsigma1_sq = (-C_ * D_) / (A * B * B);
                    float d_m_dsigma12   = (2.f * C_) / (A * B);

                    dm_dmu1[global_idx]       = d_m_dmu1;
                    dm_dsigma1_sq[global_idx] = d_m_dsigma1_sq;
                    dm_dsigma12[global_idx]   = d_m_dsigma12;
                }
            }
        }
    }
}

inline int idx4(int b, int c, int y, int x, int C, int H, int W) {
    return (((b * C + c) * H + y) * W + x);
}

kernel void fusedssim_backward(
    device const float* img1       [[ buffer(0) ]],
    device const float* img2       [[ buffer(1) ]],
    device const float* dL_dmap    [[ buffer(2) ]],
    device const float* dm_dmu1    [[ buffer(3) ]],
    device const float* dm_ds1     [[ buffer(4) ]],
    device const float* dm_ds12    [[ buffer(5) ]],
    constant float&     C1         [[ buffer(6) ]],
    constant float&     C2         [[ buffer(7) ]],
    constant int&       H          [[ buffer(8) ]],
    constant int&       W          [[ buffer(9) ]],
    constant int&       CH         [[ buffer(10) ]],
    constant int&       B          [[ buffer(11) ]],
    device float*       dL_dimg1   [[ buffer(12) ]],
    uint3 gid                      [[ thread_position_in_grid ]],
    uint3 tid                      [[ thread_position_in_threadgroup ]],
    uint3 tgroup_size              [[ threads_per_threadgroup ]],
    uint3 tgroup_pos               [[ threadgroup_position_in_grid ]],
    uint  thread_index             [[ thread_index_in_threadgroup ]]
) {
    const int pix_x = int(gid.x);
    const int pix_y = int(gid.y);
    const int pix_id = pix_y * W + pix_x;
    const int bIdx  = int(tgroup_pos.z);
    const int num_pix = H * W;

    // threadgroup storage for fused data (3 channels: v0,v1,v2)
    threadgroup float sData[3][SHARED_Y][SHARED_X];
    threadgroup float sScratch[CONV_Y][CONV_X][3];

    const int numThreads = int(tgroup_size.x) * int(tgroup_size.y);

    for (int c = 0; c < CH; ++c) {
        // read center pixel (for final accumulation)
        float p1 = 0.f, p2 = 0.f;
        if (pix_x < W && pix_y < H && bIdx < B) {
            p1 = get_pix_value(img1, bIdx, c, pix_y, pix_x, CH, H, W);
            p2 = get_pix_value(img2, bIdx, c, pix_y, pix_x, CH, H, W);
        }

        // (1) Load + fuse multiplication into sData
        // We'll parallelize load similar to CUDA: each thread loads multiple entries
            {
                const int start_y = tgroup_pos.y * int(tgroup_size.y);
                const int start_x = tgroup_pos.x * int(tgroup_size.x);
                
                int tid_global = int(tid.y) * int(tgroup_size.x) + int(tid.x);
                int warp_id = tid_global / 32;
                int lane_id = tid_global % 32;
                int num_warps = (numThreads + 31) / 32;
                
                for (int row = warp_id; row < SHARED_Y; row += num_warps) {
                    int gy = start_y + row - HALO;
                    
                    for (int col = lane_id; col < SHARED_X; col += 32) {
                        int gx = start_x + col - HALO;
                        
                        float chain = get_pix_value(dL_dmap, bIdx, c, gy, gx, CH, H, W);
                        float vmu   = get_pix_value(dm_dmu1, bIdx, c, gy, gx, CH, H, W);
                        float vs1   = get_pix_value(dm_ds1, bIdx, c, gy, gx, CH, H, W);
                        float vs12  = get_pix_value(dm_ds12, bIdx, c, gy, gx, CH, H, W);

                        sData[0][row][col] = vmu * chain;
                        sData[1][row][col] = vs1 * chain;
                        sData[2][row][col] = vs12 * chain;
                    }
                }
            }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // (2) Horizontal pass -> sScratch
        {
            int ly = int(tid.y);
            int lx = int(tid.x) + HALO;
            
            for (int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * int(tgroup_size.y);
                if (yy < CONV_Y) {
                    float accum0 = 0.f, accum1 = 0.f, accum2 = 0.f;
                    
                    for (int d = 1; d <= HALO; ++d) {
                        
                        float w = cGauss[HALO - d];
                        float left0  = sData[0][yy][lx - d];
                        float left1  = sData[1][yy][lx - d];
                        float left2  = sData[2][yy][lx - d];
                        
                        float right0 = sData[0][yy][lx + d];
                        float right1 = sData[1][yy][lx + d];
                        float right2 = sData[2][yy][lx + d];
                        
                        accum0 += (left0 + right0) * w;
                        accum1 += (left1 + right1) * w;
                        accum2 += (left2 + right2) * w;
                    }
                    
                    // center
                    {
                        float wc = cGauss[HALO];
                        float c0 = sData[0][yy][lx];
                        float c1 = sData[1][yy][lx];
                        float c2 = sData[2][yy][lx];
                        accum0 += c0 * wc;
                        accum1 += c1 * wc;
                        accum2 += c2 * wc;
                    }
                    
                    sScratch[yy][int(tid.x)][0] = accum0;
                    sScratch[yy][int(tid.x)][1] = accum1;
                    sScratch[yy][int(tid.x)][2] = accum2;
                }
                
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // (3) Vertical pass -> finalize dL/d(img1)
        if (pix_x < W && pix_y < H) {
            int ly = int(tid.y) + HALO;
            int lx = int(tid.x);
            
            float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                threadgroup float* top = sScratch[ly - d][lx];
                threadgroup float* bot = sScratch[ly + d][lx];

                sum0 += (top[0] + bot[0]) * w;
                sum1 += (top[1] + bot[1]) * w;
                sum2 += (top[2] + bot[2]) * w;
            }
            // center
            {
                float wc = cGauss[HALO];
                threadgroup float* ctr = sScratch[ly][lx];
                sum0 += ctr[0] * wc;
                sum1 += ctr[1] * wc;
                sum2 += ctr[2] * wc;
            }

            float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;
            int out_idx = bIdx * CH * num_pix + c * num_pix + pix_id;
            dL_dimg1[out_idx] = dL_dpix;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    } // channel loop
}
)FUSED_SSIM_MPS";