#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <memory>
#include <string>
#include <fstream>
#include "ssim.h" // your header with declarations

const int BLOCK_X = 16;
const int BLOCK_Y = 16;

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
        TORCH_CHECK(lib, "Failed to to create custom kernel library, error: ", err.localizedDescription.UTF8String);

        // id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:metal_source.c_str()] options:nil error:&err];
        if (err || !lib) {
            std::string msg = "Metal library compile error: ";
            if (err) {
                msg += [[err localizedDescription] UTF8String];
            }
            throw std::runtime_error(msg);
        }
        
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
        TORCH_CHECK(lib, "Failed to to create custom kernel library, error: ", err.localizedDescription.UTF8String);
        
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
