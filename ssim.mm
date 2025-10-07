#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include "ssim.h"

const int BLOCK_X = 16;
const int BLOCK_Y = 16;

static const char *MPS_KERNEL = R"FUSED_SSIM_MPS(
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

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Build a metal pipeline for function `name` from source
static id<MTLComputePipelineState> build_pipeline(id<MTLDevice> dev, id<MTLLibrary> lib, const char* name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) {
        NSLog(@"Failed to find function %s", name);
        return nil;
    }
    id<MTLComputePipelineState> pipeline = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (err) {
        NSLog(@"Pipeline error: %@", err);
        return nil;
    }
    return pipeline;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor &img1,
    torch::Tensor &img2,
    bool train
          ) {
    
    // Setup output tensors
    auto out_ssim = torch::zeros_like(img1, img1.options()).contiguous();
    auto out_dm_mu = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto out_dm_s1 = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    auto out_dm_s12 = train ? torch::zeros_like(img1) : torch::empty({0}, img1.options());
    
    @autoreleasepool{
        // ensure input shapes match
        TORCH_CHECK(img1.dim() == 4, "img1 must be BxCxHxW");
        TORCH_CHECK(img2.dim() == 4, "img2 must be BxCxHxW");
        TORCH_CHECK(img1.sizes() == img2.sizes(), "img shapes must match");
        
        int B = img1.size(0);
        int CH = img1.size(1);
        int H = img1.size(2);
        int W = img1.size(3);
        
        auto img1_contig = img1.contiguous();
        auto img2_contig = img2.contiguous();
        
        TORCH_CHECK(img1_contig.is_contiguous(),"img1 is not contiguous")
        TORCH_CHECK(img2_contig.is_contiguous(),"img2 is not contiguous")
        
        // Acquire Metal device and compile shader
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(dev, "No Metal device found");
        
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:MPS_KERNEL]
                                                                  options:nil
                                                                    error:&err];
        TORCH_CHECK(lib, "Failed to to create forward pass kernel library, error: ", err.localizedDescription.UTF8String);
        
        id<MTLComputePipelineState> pipe = build_pipeline(dev, lib, "fusedssim_forward");
        TORCH_CHECK(pipe,"Failed to create compute pipeline for fusedssim_forward");
        
        // Setup constant buffers
        id<MTLBuffer> buf_C1 = [dev newBufferWithBytes:&C1 length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_C2 = [dev newBufferWithBytes:&C2 length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_B = [dev newBufferWithBytes:&B length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_CH = [dev newBufferWithBytes:&CH length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_H = [dev newBufferWithBytes:&H length:sizeof(int) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_W = [dev newBufferWithBytes:&W length:sizeof(int) options:MTLResourceStorageModeShared];
        
        // Get torch's MPS command buffer and dispatch queue
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        
        // Add input buffers from above + tensors, and dispatch forward kernel through torch mps
        dispatch_sync(serialQueue, ^(){
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            
            // bind tensor input buffers
            [enc setBuffer:getMTLBufferStorage(img1_contig) offset:img1_contig.storage_offset() * img1_contig.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(img2_contig) offset:img2_contig.storage_offset() * img2_contig.element_size() atIndex:1];
            
            // bind constant input buffers
            [enc setBuffer:buf_C1  offset:0 atIndex:2];
            [enc setBuffer:buf_C2  offset:0 atIndex:3];
            [enc setBuffer:buf_H  offset:0 atIndex:4];
            [enc setBuffer:buf_W  offset:0 atIndex:5];
            [enc setBuffer:buf_CH offset:0 atIndex:6];
            [enc setBuffer:buf_B  offset:0 atIndex:7];
            
            // bind tensor output buffer(s)
            [enc setBuffer:getMTLBufferStorage(out_ssim)  offset:out_ssim.storage_offset() * out_ssim.element_size() atIndex:8];
            if(train){
                [enc setBuffer:getMTLBufferStorage(out_dm_mu)  offset:out_dm_mu.storage_offset() * out_dm_mu.element_size() atIndex:9];
                [enc setBuffer:getMTLBufferStorage(out_dm_s1)  offset:out_dm_s1.storage_offset() * out_dm_s1.element_size() atIndex:10];
                [enc setBuffer:getMTLBufferStorage(out_dm_s12)  offset:out_dm_s12.storage_offset() * out_dm_s12.element_size() atIndex:11];
            }
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(BLOCK_X, BLOCK_Y, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((W + BLOCK_X - 1) / BLOCK_X,
                                                      (H + BLOCK_Y - 1) / BLOCK_Y,
                                                      B);
            
            [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [enc endEncoding];
            torch::mps::commit();

        });
    }
    
    return std::make_tuple(out_ssim, out_dm_mu, out_dm_s1, out_dm_s12);
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
    
    // Setup output tensor
    torch::Tensor out_dL = torch::zeros_like(img1);
    
    @autoreleasepool{
        int B = img1.size(0);
        int CH = img1.size(1);
        int H = img1.size(2);
        int W = img1.size(3);
        
        // Store contiguous inputs
        auto img1_contig = img1.contiguous();
        auto img2_contig = img2.contiguous();
        auto dL_dmap_contig = dL_dmap.contiguous();
        auto dm_dmu1_contig = dm_dmu1.contiguous();
        auto dm_dsigma1_sq_contig = dm_dsigma1_sq.contiguous();
        auto dm_dsigma12_contig = dm_dsigma12.contiguous();
        
        // Acquire Metal device and compile shader
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(dev,"No Metal device found");
        
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:MPS_KERNEL]
                                                                  options:nil
                                                                    error:&err];
        TORCH_CHECK(lib, "Failed to to create backward pass kernel library, error: ", err.localizedDescription.UTF8String);
        
        id<MTLComputePipelineState> pipe = build_pipeline(dev, lib, "fusedssim_backward");
        TORCH_CHECK(pipe,"Failed to create pipeline for backward");
    
        // Setup constant buffers
        id<MTLBuffer> b_C1 = [dev newBufferWithBytes:&C1 length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_C2 = [dev newBufferWithBytes:&C2 length:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_H  = [dev newBufferWithBytes:&H  length:sizeof(int)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_W  = [dev newBufferWithBytes:&W  length:sizeof(int)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_CH = [dev newBufferWithBytes:&CH length:sizeof(int)   options:MTLResourceStorageModeShared];
        id<MTLBuffer> b_B  = [dev newBufferWithBytes:&B  length:sizeof(int)   options:MTLResourceStorageModeShared];
        
        // Get torch's MPS command buffer and dispatch queue
        id<MTLCommandBuffer> cb = torch::mps::get_command_buffer();
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        
        // Add input buffers from above + tensors, and dispatch backward kernel through torch mps
        dispatch_sync(serialQueue, ^(){
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipe];
            
            // bind tensor input buffers
            [enc setBuffer:getMTLBufferStorage(img1_contig) offset:img1_contig.storage_offset() * img1_contig.element_size() atIndex:0];
            [enc setBuffer:getMTLBufferStorage(img2_contig) offset:img2_contig.storage_offset() * img2_contig.element_size() atIndex:1];
            [enc setBuffer:getMTLBufferStorage(dL_dmap_contig) offset:dL_dmap_contig.storage_offset() * dL_dmap_contig.element_size() atIndex:2];
            [enc setBuffer:getMTLBufferStorage(dm_dmu1_contig) offset:dm_dmu1_contig.storage_offset() * dm_dmu1_contig.element_size() atIndex:3];
            [enc setBuffer:getMTLBufferStorage(dm_dsigma1_sq_contig) offset:dm_dsigma1_sq_contig.storage_offset() * dm_dsigma1_sq_contig.element_size() atIndex:4];
            [enc setBuffer:getMTLBufferStorage(dm_dsigma12_contig) offset:dm_dsigma12_contig.storage_offset() * dm_dsigma12_contig.element_size() atIndex:5];
            
            // bind constant input buffers
            [enc setBuffer:b_C1 offset:0 atIndex:6];
            [enc setBuffer:b_C2 offset:0 atIndex:7];
            [enc setBuffer:b_H offset:0 atIndex:8];
            [enc setBuffer:b_W offset:0 atIndex:9];
            [enc setBuffer:b_CH offset:0 atIndex:10];
            [enc setBuffer:b_B offset:0 atIndex:11];
            
            // bind tensor output buffer
            [enc setBuffer:getMTLBufferStorage(out_dL) offset:out_dL.storage_offset() * out_dL.element_size() atIndex:12];
            
            MTLSize threadsPerThreadgroup = MTLSizeMake(BLOCK_X, BLOCK_Y, 1);
            MTLSize threadgroupsPerGrid = MTLSizeMake((W + BLOCK_X - 1) / BLOCK_X,
                                                      (H + BLOCK_Y - 1) / BLOCK_Y,
                                                      B);
            [enc dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
            [enc endEncoding];
            torch::mps::commit();
        });
    }
    
    return out_dL;
}
