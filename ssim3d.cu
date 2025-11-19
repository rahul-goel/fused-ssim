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
#define BLOCK_X 8
#define BLOCK_Y 4
#define BLOCK_Z 4
#define HALO    5

#define SHARED_X (BLOCK_X + 2 * HALO)
#define SHARED_Y (BLOCK_Y + 2 * HALO)
#define SHARED_Z (BLOCK_Z + 2 * HALO)

// For partial results after horizontal pass
#define CONV_X BLOCK_X
#define CONV_Y SHARED_Y
#define CONV_Z SHARED_Z 

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

// struct Xconv5 {
//     float sumX;
//     float sumX2;
//     float sumY;
//     float sumY2;
//     float sumXY;
// };

__device__ __forceinline__ int tile_idx(int z, int y, int x, int channel) {
    return ((((z * SHARED_Y) + y) * SHARED_X) + x) * 2 + channel;
}

__device__ __forceinline__ int xconv_idx(int z, int y, int x, int stat) {
    return ((((z * CONV_Y) + y) * CONV_X) + x) * 5 + stat;
}

__device__ __forceinline__ void accumulate_xaxis(
    float* sTile,
    int lz, int ly, int lx, float out[5]
) {
    out[0] = out[1] = out[2] = out[3] = out[4] = 0.f;

#pragma unroll
    for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float Xleft  = sTile[tile_idx(lz, ly, lx - d, 0)];
        float Yleft  = sTile[tile_idx(lz, ly, lx - d, 1)];
        float Xright = sTile[tile_idx(lz, ly, lx + d, 0)];
        float Yright = sTile[tile_idx(lz, ly, lx + d, 1)];

        out[0]  += (Xleft + Xright) * w;
        out[1] += ((Xleft * Xleft) + (Xright * Xright)) * w;
        out[2]  += (Yleft + Yright) * w;
        out[3] += ((Yleft * Yleft) + (Yright * Yright)) * w;
        out[4] += ((Xleft * Yleft) + (Xright * Yright)) * w;
    }

    float centerX = sTile[tile_idx(lz, ly, lx, 0)];
    float centerY = sTile[tile_idx(lz, ly, lx, 1)];
    float wc = cGauss[HALO];
    out[0] += centerX * wc;
    out[1] += (centerX * centerX) * wc;
    out[2] += centerY * wc;
    out[3] += (centerY * centerY) * wc;
    out[4] += (centerX * centerY) * wc;
}

__device__ __forceinline__ void accumulate_yaxis(
    float* sTile,
    int lz, int ly, int lx, float out[5]
) {
    out[0] = out[1] = out[2] = out[3] = out[4] = 0.f;
#pragma unroll
    for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float* top = &sTile[xconv_idx(lz, ly - d, lx,0)];
        float* bot = &sTile[xconv_idx(lz, ly + d, lx,0)];

        out[0] += (top[0] + bot[0]) * w;
        out[1] += (top[1] + bot[1]) * w;
        out[2] += (top[2] + bot[2]) * w;
        out[3] += (top[3] + bot[3]) * w;
        out[4] += (top[4] + bot[4]) * w;
    }   
        float* center = &sTile[xconv_idx(lz, ly, lx, 0)];
        float wc = cGauss[HALO];
        out[0]  += center[0] * wc;
        out[1] += (center[0] * center[0]) * wc;
        out[2] += center[1] * wc;
        out[3] += (center[1] * center[1]) * wc;
        out[4] += (center[0] * center[1]) * wc;
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
    float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();
    // const int bIdx   = block.group_index().z;  // batch index
    const int pix_z  = block.group_index().z * BLOCK_Z + block.thread_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_z * W * H + pix_y * W + pix_x;
    const int num_pix = H * W * D;

    //Initialize one shared memory block to serve both the image and intermediate sums
    __shared__ float sTile[CONV_Z * CONV_Y * CONV_X * 5]; //CONV len requirements are larger than SHARED len

    // // Shared memory for the tile (img1, img2)
    // __shared__ float sTile[SHARED_Y][SHARED_X][2];
    // // After horizontal pass, store partial sums here
    // // xconv[y][x] -> (sumX, sumX^2, sumY, sumY^2, sumXY)
    // __shared__ float xconv[CONV_Y][CONV_X][5];

    // Each block processes B x C sub-batches. We loop over batches and channels:
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < CH; ++c) {
            // ------------------------------------------------------------
            // 1) Load (img1, img2) tile + halo into shared memory
            // ------------------------------------------------------------
            {
                const int tileSize = SHARED_Z * SHARED_Y * SHARED_X;
                const int threads = BLOCK_Z * BLOCK_Y * BLOCK_X;
                const int steps = (tileSize + threads - 1) / threads;
                
                const int tileStartZ = block.group_index().z * BLOCK_Z;
                const int tileStartY = block.group_index().y * BLOCK_Y;
                const int tileStartX = block.group_index().x * BLOCK_X;

                for (int s = 0; s < steps; ++s) {
                    int tid = s * threads + block.thread_rank(); //thread id
                    if (tid < tileSize) {
                        int local_z = tid / (SHARED_Y * SHARED_X);
                        int rem = tid % (SHARED_Y * SHARED_X);
                        int local_y = rem / SHARED_X;
                        int local_x = rem % SHARED_X;
                        int gz = tileStartZ + local_z - HALO;
                        int gy = tileStartY + local_y - HALO;
                        int gx = tileStartX + local_x - HALO;

                        float X = get_pix_value(img1, b, c, gz, gy, gx, CH, H, W, D);
                        float Y = get_pix_value(img2, b, c, gz, gy, gx, CH, H, W, D);

                        sTile[tile_idx(local_z, local_y, local_x, 0)] = X;
                        sTile[tile_idx(local_z, local_y, local_x, 1)] = Y;
                    }
                }
            }
            block.sync();

            // ------------------------------------------------------------
            // 2) X axis convolution (11x1) in shared memory
            //    We'll accumulate symmetrical pairs around center.
            // ------------------------------------------------------------
             
            int lz = threadIdx.z;
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;  // skip left halo

            int ly2 = ly + BLOCK_Y;      // second Y row in same warp
            int ly3 = ly + 2*BLOCK_Y;    // third Y row in same warp
            int ly4 = ly + 3*BLOCK_Y;    // fourth Y row in same warp

            int lz2 = lz + BLOCK_Z;      // second Z row in same warp
            int lz3 = lz + 2*BLOCK_Z;    // third Z row in same warp
            int lz4 = lz + 3*BLOCK_Z;    // fourth Z row in same warp

            float *dst; // pointer to write xconv results

            float xconv[5], xconv_y2[5], xconv_y3[5], xconv_y4[5], xconv_z2[5], xconv_z3[5], xconv_z4[5];

            accumulate_xaxis(sTile, lz, ly, lx, xconv);
            
            // Handle second and third y pass (here due to smaller block size, it always happens)
            accumulate_xaxis(sTile, lz, ly2, lx, xconv_y2);
            accumulate_xaxis(sTile, lz, ly3, lx, xconv_y3);
            

            // Possibly handle fourth Y row in same warp
            if (ly4 < CONV_Y) {
                accumulate_xaxis(sTile, lz, ly4, lx, xconv_y4);
            }

            // Handle second and third Z row in same warp
            accumulate_xaxis(sTile, lz2, ly, lx, xconv_z2);
            accumulate_xaxis(sTile, lz3, ly, lx, xconv_z3);
            // Possibly handle fourth Z row in same warp
            if (lz4 < CONV_Z) {
                accumulate_xaxis(sTile, lz4, ly, lx, xconv_z4);
            }

            block.sync();

            // Overwrite sTile with xconv from X axis pass
            dst = &sTile[xconv_idx(lz,ly,threadIdx.x, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv[i];
            
            dst = &sTile[xconv_idx(lz,ly2,threadIdx.x, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_y2[i];

            dst = &sTile[xconv_idx(lz,ly3,threadIdx.x, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_y3[i];

            if (ly4 < CONV_Y) {
                dst = &sTile[xconv_idx(lz,ly4,threadIdx.x, 0)];
                #pragma unroll 
                for (int i = 0; i < 5; ++i) dst[i] = xconv_y4[i];
            }

            dst = &sTile[xconv_idx(lz2,ly,threadIdx.x, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_z2[i];

            dst = &sTile[xconv_idx(lz3,ly,threadIdx.x, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_z3[i];

            if (lz4 < CONV_Z) {
                dst = &sTile[xconv_idx(lz4,ly,threadIdx.x, 0)];
                #pragma unroll 
                for (int i = 0; i < 5; ++i) dst[i] = xconv_z4[i];
            }
            
            
            block.sync();
            // ------------------------------------------------------------
            // 3) Y axis convolution (1x11)
            // ------------------------------------------------------------   
            // lz = threadIdx.z;
            ly = threadIdx.y + HALO;
            lx = threadIdx.x; 

            accumulate_yaxis(sTile, lz, ly, lx, xconv);

            // Handle second and third Z row in same warp
            accumulate_yaxis(sTile, lz2, ly, lx, xconv_z2);
            accumulate_yaxis(sTile, lz3, ly, lx, xconv_z3);
            // Possibly handle fourth Z row in same warp
            if (lz4 < CONV_Z) {
                accumulate_yaxis(sTile, lz4, ly, lx, xconv_z4);
            }

            block.sync();

            // Overwrite sTile with xconv from Y axis pass
            
            dst = &sTile[xconv_idx(lz,threadIdx.y,lx, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv[i];

            dst = &sTile[xconv_idx(lz2,threadIdx.y,lx, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_z2[i];

            dst = &sTile[xconv_idx(lz3,threadIdx.y,lx, 0)];
            #pragma unroll 
            for (int i = 0; i < 5; ++i) dst[i] = xconv_z3[i];
         
            if (lz4 < CONV_Z) {
                dst = &sTile[xconv_idx(lz4,threadIdx.y,lx, 0)];
                #pragma unroll 
                for (int i = 0; i < 5; ++i) dst[i] = xconv_z4[i];
            }
            
            block.sync();

            // ------------------------------------------------------------
            // 3) Z axis convolution (1x11) + final SSIM
            // ------------------------------------------------------------
            
            lz = threadIdx.z + HALO;
            ly = threadIdx.y;
            // lx = threadIdx.x;
            xconv[0] = xconv[1] = xconv[2] = xconv[3] = xconv[4] = 0.f;

#pragma unroll
            for (int d = 1; d <= HALO; ++d) {
                float w = cGauss[HALO - d];
                float* top = &sTile[xconv_idx(lz - d, ly, lx, 0)];
                float* bot = &sTile[xconv_idx(lz + d, ly, lx,0)];

                xconv[0] += (top[0] + bot[0]) * w;
                xconv[1] += (top[1] + bot[1]) * w;
                xconv[2] += (top[2] + bot[2]) * w;
                xconv[3] += (top[3] + bot[3]) * w;
                xconv[4] += (top[4] + bot[4]) * w;
            }
            // center
            {
                float wC = cGauss[HALO];
                float* ctr = &sTile[xconv_idx(lz, ly, lx, 0)];
                xconv[0] += ctr[0] * wC;
                xconv[1] += ctr[1] * wC;
                xconv[2] += ctr[2] * wC;
                xconv[3] += ctr[3] * wC;
                xconv[4] += ctr[4] * wC;
            }

            if (pix_x < W && pix_y < H && pix_z < D) {
                float mu1 = xconv[0];
                float mu2 = xconv[2];
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;

                float sigma1_sq = xconv[1] - mu1_sq;
                float sigma2_sq = xconv[3] - mu2_sq;
                float sigma12   = xconv[4] - mu1 * mu2;

                float A = mu1_sq + mu2_sq + C1;
                float B = sigma1_sq + sigma2_sq + C2;
                float C_ = 2.f * mu1 * mu2 + C1;
                float D_ = 2.f * sigma12 + C2;

                float val = (C_ * D_) / (A * B);

                int global_idx = b * CH * num_pix + c * num_pix + pix_id;
                ssim_map[global_idx] = val;

                if (dm_dmu1) {
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

__device__ __forceinline__ int sdata_idx(int stat, int z, int y, int x) {
    return (stat * SHARED_Z * SHARED_Y * SHARED_X) + (z * SHARED_Y * SHARED_X) + (y * SHARED_X) + x;
}

__device__ __forceinline__ int scratch_idx(int z, int y, int x, int stat) {
    return (((z * CONV_Y) + y) * CONV_X + x) * 3 + stat;
}

__device__ __forceinline__ void accumulate_xaxis_back(
    float* sData,
    int lz, int ly, int lx, float out[3]
) {
    out[0] = out[1] = out[2] = 0.f;
#pragma unroll
    for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float left0  = sData[sdata_idx(0, lz, ly, lx - d)];
        float left1  = sData[sdata_idx(1, lz, ly, lx - d)];
        float left2  = sData[sdata_idx(2, lz, ly, lx - d)];

        float right0 = sData[sdata_idx(0, lz, ly, lx + d)];
        float right1 = sData[sdata_idx(1, lz, ly, lx + d)];
        float right2 = sData[sdata_idx(2, lz, ly, lx + d)];

        out[0] += (left0 + right0) * w;
        out[1] += (left1 + right1) * w;
        out[2] += (left2 + right2) * w;
    }
    // center
    {
        float wc = cGauss[HALO];
        float c0 = sData[sdata_idx(0, lz, ly, lx)];
        float c1 = sData[sdata_idx(1, lz, ly, lx)];
        float c2 = sData[sdata_idx(2, lz, ly, lx)];
        out[0] += c0 * wc;
        out[1] += c1 * wc;
        out[2] += c2 * wc;
    }
}

__device__ __forceinline__ void accumulate_yaxis_back(
    float* sData,
    int lz, int ly, int lx, float out[3]
) {
    out[0] = out[1] = out[2] = 0.f;
#pragma unroll
    for (int d = 1; d <= HALO; ++d) {
        float w = cGauss[HALO - d];
        float* top0 = &sData[scratch_idx(lz, ly - d, lx, 0)];
        float* bot0 = &sData[scratch_idx(lz, ly + d, lx, 0)];

        out[0] += (top0[0] + bot0[0]) * w;
        out[1] += (top0[1] + bot0[1]) * w;
        out[2] += (top0[2] + bot0[2]) * w;
    }   
        float* center = &sData[scratch_idx(lz, ly, lx, 0)];
        float wc = cGauss[HALO];
        out[0]  += center[0] * wc;
        out[1] += center[1] * wc;
        out[2] += center[2] * wc;
}


// ------------------------------------------
// Backward Kernel: Apply chain rule to get
//    dL/d(img1) from partial derivatives
//    (dm_dmu1, dm_dsigma1_sq, dm_dsigma12)
//    and dL/dmap (the gradient from above).
// ------------------------------------------
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
    const float* __restrict__ dm_dsigma12
) {
    auto block = cg::this_thread_block();

    const int pix_z  = block.group_index().z * BLOCK_Z + block.thread_index().z;
    const int pix_y  = block.group_index().y * BLOCK_Y + block.thread_index().y;
    const int pix_x  = block.group_index().x * BLOCK_X + block.thread_index().x;
    const int pix_id = pix_z * W * H + pix_y * W + pix_x;
    const int num_pix = H * W * D;
    // const int bIdx   = block.group_index().z;

    // Shared memory for the fused data:
    // [0]: dm_dmu1*dL, [1]: dm_dsigma1_sq*dL, [2]: dm_dsigma12*dL
    // __shared__ float sData[3][SHARED_Y][SHARED_X];
    // __shared__ float sScratch[CONV_Y][CONV_X][3];
    // Initialize one shared memory block to serve both the fused data and scratch
    __shared__ float sData[3 * SHARED_Z * SHARED_Y * SHARED_X];

    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < CH; ++c) {
            float p1 = 0.f, p2 = 0.f;
            if (pix_x < W && pix_y < H) {
                p1 = get_pix_value(img1, b, c, pix_z, pix_y, pix_x, CH, H, W, D);
                p2 = get_pix_value(img2, b, c, pix_z, pix_y, pix_x, CH, H, W, D);
            }

            // (1) Load + fuse multiplication
            {   
                const int start_z = block.group_index().z * BLOCK_Z;
                const int start_y = block.group_index().y * BLOCK_Y;
                const int start_x = block.group_index().x * BLOCK_X;

                int tid = threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
                int warp_id = tid / 32; //warp index within block
                int lane_id = tid % 32; //lane index within warp
                int totalThreads = BLOCK_X * BLOCK_Y * BLOCK_Z; //total threads per block
                int num_warps = (totalThreads + 31) / 32;
                
                for (int depth = warp_id; depth < SHARED_Z; depth += num_warps) {
                    int gz = start_z + depth - HALO;
                    for (int row = 0; row < SHARED_Y; ++row) {
                        int gy = start_y + row - HALO;
                        for (int col = lane_id; col < SHARED_X; col += 32) {
                            int gx = start_x + col - HALO;

                            float chain = get_pix_value(dL_dmap,      b, c, gz, gy, gx, CH, H, W, D);
                            float vmu   = get_pix_value(dm_dmu1,      b, c, gz, gy, gx, CH, H, W, D);
                            float vs1   = get_pix_value(dm_dsigma1_sq,b, c, gz, gy, gx, CH, H, W, D);
                            float vs12  = get_pix_value(dm_dsigma12,  b, c, gz, gy, gx, CH, H, W, D);

                            sData[sdata_idx(0, depth, row, col)] = vmu  * chain;
                            sData[sdata_idx(1, depth, row, col)] = vs1  * chain;
                            sData[sdata_idx(2, depth, row, col)] = vs12 * chain;
                        }
                    }
                }
            }
            block.sync();

            //------------------------------------------------------------
            // (2) X axis pass
            //------------------------------------------------------------
            int lz = threadIdx.z;
            int ly = threadIdx.y;
            int lx = threadIdx.x + HALO;
            
            
            int ly2 = ly + BLOCK_Y;      // second Y row in same warp
            int ly3 = ly + 2*BLOCK_Y;    // third Y row in same warp
            int ly4 = ly + 3*BLOCK_Y;    // fourth Y row in same warp

            int lz2 = lz + BLOCK_Z;      // second Z row in same warp
            int lz3 = lz + 2*BLOCK_Z;    // third Z row in same warp
            int lz4 = lz + 3*BLOCK_Z;    // fourth Z row in same warp

            float *dst; // pointer to write sScratch results

            float sScratch[3], sScratch_y2[3], sScratch_y3[3], sScratch_y4[3], sScratch_z2[3], sScratch_z3[3], sScratch_z4[3];

            accumulate_xaxis_back(sData, lz, ly, lx, sScratch);
            // Handle second and third y pass (here due to smaller block size, it always happens)
            accumulate_xaxis_back(sData, lz, ly2, lx, sScratch_y2);
            accumulate_xaxis_back(sData, lz, ly3, lx, sScratch_y3);
            // Possibly handle fourth Y row in same warp
            if (ly4 < CONV_Y) {
                accumulate_xaxis_back(sData, lz, ly4, lx, sScratch_y4);
            }
            // Handle second and third Z row in same warp
            accumulate_xaxis_back(sData, lz2, ly, lx, sScratch_z2);
            accumulate_xaxis_back(sData, lz3, ly, lx, sScratch_z3);
            // Possibly handle fourth Z row in same warp
            if (lz4 < CONV_Z) {
                accumulate_xaxis_back(sData, lz4, ly, lx, sScratch_z4);
            }

            block.sync();

            // Overwrite sData with X axis pass sScratch results

            dst = &sData[scratch_idx(lz,ly,threadIdx.x, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch[i];

            dst = &sData[scratch_idx(lz,ly2,threadIdx.x, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_y2[i];
            dst = &sData[scratch_idx(lz,ly3,threadIdx.x, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_y3[i];
            if (ly4 < CONV_Y) {
                dst = &sData[scratch_idx(lz,ly4,threadIdx.x, 0)];
                #pragma unroll
                for (int i = 0; i < 3; ++i) dst[i] = sScratch_y4[i];
            }

            dst = &sData[scratch_idx(lz2,ly,threadIdx.x, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_z2[i];
            dst = &sData[scratch_idx(lz3,ly,threadIdx.x, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_z3[i];
            if (lz4 < CONV_Z) {
                dst = &sData[scratch_idx(lz4,ly,threadIdx.x, 0)];
                #pragma unroll
                for (int i = 0; i < 3; ++i) dst[i] = sScratch_z4[i];
            }

            block.sync();

            //----------------------------------------------------------
            // (2) Y axis pass
            //----------------------------------------------------------
            ly = threadIdx.y + HALO;
            lx = threadIdx.x; 

            accumulate_yaxis_back(sData, lz, ly, lx, sScratch);
            // Handle second and third Z row in same warp
            accumulate_yaxis_back(sData, lz2, ly, lx, sScratch_z2);
            accumulate_yaxis_back(sData, lz3, ly, lx, sScratch_z3);
            // Possibly handle fourth Z row in same warp
            if (lz4 < CONV_Z) {
                accumulate_yaxis_back(sData, lz4, ly, lx, sScratch_z4);
            }

            block.sync();

            // Overwrite sData with Y axis pass sScratch results
            dst = &sData[scratch_idx(lz,threadIdx.y,lx, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch[i];

            dst = &sData[scratch_idx(lz2,threadIdx.y,lx, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_z2[i];
            dst = &sData[scratch_idx(lz3,threadIdx.y,lx, 0)];
            #pragma unroll
            for (int i = 0; i < 3; ++i) dst[i] = sScratch_z3[i];
            if (lz4 < CONV_Z) {
                dst = &sData[scratch_idx(lz4,threadIdx.y,lx, 0)];
                #pragma unroll
                for (int i = 0; i < 3; ++i) dst[i] = sScratch_z4[i];
            }

            block.sync();

            //----------------------------------------------------------
            // (3) Z axis pass -> finalize dL/d(img1)
            //----------------------------------------------------------
            if (pix_x < W && pix_y < H && pix_z < D) {
                int lz = threadIdx.z + HALO;
                int ly = threadIdx.y;

                float sum0 = 0.f, sum1 = 0.f, sum2 = 0.f;

    #pragma unroll
                for (int d = 1; d <= HALO; ++d) {
                    float w = cGauss[HALO - d];
                    float* top = &sData[scratch_idx(lz - d, ly, lx, 0)];
                    float* bot = &sData[scratch_idx(lz + d, ly, lx, 0)];

                    sum0 += (top[0] + bot[0]) * w;
                    sum1 += (top[1] + bot[1]) * w;
                    sum2 += (top[2] + bot[2]) * w;
                }
                // center
                {
                    float wc = cGauss[HALO];
                    float* ctr = &sData[scratch_idx(lz, ly, lx, 0)];
                    sum0 += ctr[0] * wc;
                    sum1 += ctr[1] * wc;
                    sum2 += ctr[2] * wc;
                }

                // final accumulation
                float dL_dpix = sum0 + (2.f * p1) * sum1 + (p2) * sum2;

                int out_idx = b * CH * num_pix + c * num_pix + pix_id;
                dL_dimg1[out_idx] = dL_dpix;
            }
            block.sync();
        }
    }
}


// ------------------------------------------
// PyTorch Interface (Forward)
//   Returns (ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12).
//   If train=false, derivative Tensors are empty.
// ------------------------------------------
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

    // Launch config
    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              (D + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    // Output SSIM map
    auto ssim_map = torch::zeros_like(img1, img1.options()).contiguous();

    // Optionally allocate derivative Tensors
    auto dm_dmu1       = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());

    fusedssim3dCUDA<<<grid, block>>>(
        H, W, D, B, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        ssim_map.data_ptr<float>(),
        train ? dm_dmu1.data_ptr<float>()       : nullptr,
        train ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
        train ? dm_dsigma12.data_ptr<float>()   : nullptr
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
    int H  = img1.size(2);
    int W  = img1.size(3);

    auto dL_dimg1 = torch::zeros_like(img1);

    dim3 grid((W + BLOCK_X - 1) / BLOCK_X,
              (H + BLOCK_Y - 1) / BLOCK_Y,
              (D + BLOCK_Z - 1) / BLOCK_Z);
    dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

    fusedssim_backward3dCUDA<<<grid, block>>>(
        H, W, D, B, CH, C1, C2,
        img1.contiguous().data_ptr<float>(),
        img2.contiguous().data_ptr<float>(),
        dL_dmap.contiguous().data_ptr<float>(),
        dL_dimg1.data_ptr<float>(),
        dm_dmu1.contiguous().data_ptr<float>(),
        dm_dsigma1_sq.contiguous().data_ptr<float>(),
        dm_dsigma12.contiguous().data_ptr<float>()
    );

    return dL_dimg1;
}