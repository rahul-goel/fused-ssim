// PyTorch MPS extension wrapper for Halide-generated SSIM
// Zero-copy integration using Metal buffers

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <torch/extension.h>
#include "ssim_halide.h"
#include "HalideBuffer.h"
#include "HalideRuntimeMetal.h"

#include <tuple>
#include <stdexcept>

// Helper to get MTLBuffer from PyTorch MPS tensor
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

// Helper to create a Halide buffer wrapping PyTorch MPS tensor
// Returns the retained MTLBuffer pointer that must be released with CFRelease
CFTypeRef wrap_mps_tensor_for_halide(
    const torch::Tensor& tensor,
    halide_buffer_t* halide_buf
) {
    // printf("[wrap_mps_tensor] Starting tensor wrapping\n");
    // Get dimensions (NCHW -> WHC for Halide)
    int batch = tensor.size(0);
    int channels = tensor.size(1);
    int height = tensor.size(2);
    int width = tensor.size(3);
    // printf("[wrap_mps_tensor] Dimensions: batch=%d, channels=%d, height=%d, width=%d\n",
    //        batch, channels, height, width);

    // Only support batch size 1 for now
    if (batch != 1) {
        throw std::runtime_error("Halide SSIM currently only supports batch size 1");
    }

    // Set up Halide buffer dimensions: WHC layout
    halide_buf->dimensions = 3;
    halide_buf->dim = (halide_dimension_t*)malloc(3 * sizeof(halide_dimension_t));

    // x dimension (width)
    halide_buf->dim[0].min = 0;
    halide_buf->dim[0].extent = width;
    halide_buf->dim[0].stride = 1;

    // y dimension (height)
    halide_buf->dim[1].min = 0;
    halide_buf->dim[1].extent = height;
    halide_buf->dim[1].stride = width;

    // c dimension (channels)
    halide_buf->dim[2].min = 0;
    halide_buf->dim[2].extent = channels;
    halide_buf->dim[2].stride = height * width;

    halide_buf->type.code = halide_type_float;
    halide_buf->type.bits = 32;
    halide_buf->type.lanes = 1;

    // Get the Metal buffer from PyTorch
    // printf("[wrap_mps_tensor] Getting Metal buffer from PyTorch\n");
    id<MTLBuffer> mtl_buffer = getMTLBufferStorage(tensor);
    // printf("[wrap_mps_tensor] Got Metal buffer: %p\n", (__bridge void*)mtl_buffer);

    // Calculate offset in bytes
    size_t offset = tensor.storage_offset() * tensor.element_size();
    // printf("[wrap_mps_tensor] Offset: %zu bytes\n", offset);

    // Set the host pointer to nullptr (using device memory)
    halide_buf->host = nullptr;

    // Initialize device fields
    halide_buf->device = 0;
    halide_buf->device_interface = halide_metal_device_interface();

    // Wrap the Metal buffer for Halide
    // Cast MTLBuffer to uint64_t as expected by halide_metal_wrap_buffer
    // printf("[wrap_mps_tensor] Wrapping Metal buffer for Halide\n");
    uint64_t buffer_handle = (uint64_t)(__bridge void*)mtl_buffer;
    int result = halide_metal_wrap_buffer(nullptr, halide_buf, buffer_handle);
    // printf("[wrap_mps_tensor] halide_metal_wrap_buffer result: %d\n", result);
    if (result != 0) {
        free(halide_buf->dim);
        throw std::runtime_error("Failed to wrap Metal buffer for Halide");
    }

    // Apply offset if needed
    // Note: This adjusts the device pointer within the buffer
    if (offset > 0) {
        halide_buf->device += offset;
    }

    // Retain the buffer so it doesn't get deallocated
    // CFBridgingRetain retains and returns a CF-owned reference (manual management)
    // printf("[wrap_mps_tensor] Retaining buffer and returning\n");
    CFTypeRef retained = CFBridgingRetain(mtl_buffer);
    // printf("[wrap_mps_tensor] Successfully wrapped tensor\n");
    return retained;
}

// Clean up Halide buffer
void cleanup_halide_buffer(halide_buffer_t* buf, CFTypeRef mtl_buf) {
    // printf("[cleanup] Starting buffer cleanup\n");
    halide_metal_detach_buffer(nullptr, buf);
    // printf("[cleanup] Detached buffer from Halide\n");
    // Release the manually-retained buffer
    if (mtl_buf) {
        CFRelease(mtl_buf);
        // printf("[cleanup] Released Metal buffer\n");
    }
    free(buf->dim);
    // printf("[cleanup] Freed dimension array\n");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim_mps(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train
) {
    // printf("\n[fusedssim_mps] ========== Starting fusedssim_mps ==========\n");
    // printf("[fusedssim_mps] C1=%f, C2=%f, train=%d\n", C1, C2, train);

    // Validate inputs
    // printf("[fusedssim_mps] Validating inputs\n");
    if (img1.dim() != 4 || img2.dim() != 4) {
        throw std::runtime_error("Input tensors must be 4D (NCHW)");
    }
    if (img1.dtype() != torch::kFloat32 || img2.dtype() != torch::kFloat32) {
        throw std::runtime_error("Input tensors must be float32");
    }
    if (img1.sizes() != img2.sizes()) {
        throw std::runtime_error("Input tensors must have the same shape");
    }
    if (img1.device().type() != torch::kMPS || img2.device().type() != torch::kMPS) {
        throw std::runtime_error("Input tensors must be on MPS device");
    }

    // Make tensors contiguous
    // printf("[fusedssim_mps] Making tensors contiguous\n");
    auto img1_contig = img1.contiguous();
    auto img2_contig = img2.contiguous();
    // printf("[fusedssim_mps] Tensors are contiguous\n");

    @autoreleasepool {
        // printf("[fusedssim_mps] Entered autorelease pool\n");

        // Synchronize PyTorch MPS before wrapping buffers
        // This ensures any pending PyTorch operations complete before Halide accesses the buffers
        // printf("[fusedssim_mps] Synchronizing PyTorch MPS\n");
        torch::mps::synchronize();
        // printf("[fusedssim_mps] PyTorch MPS synchronized\n");

        // Let Halide create its own Metal context instead of sharing PyTorch's
        // This avoids synchronization issues between the two runtimes
        // printf("[fusedssim_mps] Letting Halide create its own Metal context\n");

        // Allocate output tensors on MPS
        // Note: Halide always computes all outputs, so we must allocate all buffers
        // printf("[fusedssim_mps] Allocating output tensors\n");
        torch::Tensor ssim_out = torch::empty_like(img1_contig);
        torch::Tensor dm_dmu1_out = torch::empty_like(img1_contig);
        torch::Tensor dm_dsigma1_sq_out = torch::empty_like(img1_contig);
        torch::Tensor dm_dsigma12_out = torch::empty_like(img1_contig);
        // printf("[fusedssim_mps] Output tensors allocated\n");

        // Create Halide buffers wrapping PyTorch MPS tensors
        halide_buffer_t buf1 = {0};
        halide_buffer_t buf2 = {0};
        halide_buffer_t ssim_buf = {0};
        halide_buffer_t dm_dmu1_buf = {0};
        halide_buffer_t dm_dsigma1_sq_buf = {0};
        halide_buffer_t dm_dsigma12_buf = {0};

        CFTypeRef mtl_buf1 = nullptr;
        CFTypeRef mtl_buf2 = nullptr;
        CFTypeRef mtl_buf_ssim = nullptr;
        CFTypeRef mtl_buf_dmu1 = nullptr;
        CFTypeRef mtl_buf_dsigma1 = nullptr;
        CFTypeRef mtl_buf_dsigma12 = nullptr;

        try {
            // Wrap input and output tensors
            // printf("[fusedssim_mps] Wrapping input tensor 1\n");
            mtl_buf1 = wrap_mps_tensor_for_halide(img1_contig, &buf1);
            // printf("[fusedssim_mps] Wrapping input tensor 2\n");
            mtl_buf2 = wrap_mps_tensor_for_halide(img2_contig, &buf2);
            // printf("[fusedssim_mps] Wrapping SSIM output tensor\n");
            mtl_buf_ssim = wrap_mps_tensor_for_halide(ssim_out, &ssim_buf);
            // printf("[fusedssim_mps] Wrapping dmu1 output tensor\n");
            mtl_buf_dmu1 = wrap_mps_tensor_for_halide(dm_dmu1_out, &dm_dmu1_buf);
            // printf("[fusedssim_mps] Wrapping dsigma1_sq output tensor\n");
            mtl_buf_dsigma1 = wrap_mps_tensor_for_halide(dm_dsigma1_sq_out, &dm_dsigma1_sq_buf);
            // printf("[fusedssim_mps] Wrapping dsigma12 output tensor\n");
            mtl_buf_dsigma12 = wrap_mps_tensor_for_halide(dm_dsigma12_out, &dm_dsigma12_buf);
            // printf("[fusedssim_mps] All tensors wrapped successfully\n");

            // Call Halide-generated function directly
            // Halide will manage its own command buffer submission on the Metal queue
            // printf("[fusedssim_mps] Calling fused_ssim_forward\n");
            // fflush(stdout);
            int err = fused_ssim_forward(
                const_cast<halide_buffer_t*>(&buf1),
                const_cast<halide_buffer_t*>(&buf2),
                C1,
                C2,
                const_cast<halide_buffer_t*>(&ssim_buf),
                const_cast<halide_buffer_t*>(&dm_dmu1_buf),
                const_cast<halide_buffer_t*>(&dm_dsigma1_sq_buf),
                const_cast<halide_buffer_t*>(&dm_dsigma12_buf)
            );
            // printf("[fusedssim_mps] fused_ssim_forward returned: %d\n", err);
            // fflush(stdout);

            if (err != 0) {
                throw std::runtime_error("Halide fused_ssim_forward failed with error code: " + std::to_string(err));
            }
            // printf("[fusedssim_mps] Halide kernel completed successfully\n");

            // Note: Halide manages its own command buffer submission
            // We don't need to commit PyTorch's command buffer
            // printf("[fusedssim_mps] Halide handles its own command buffer submission\n");

            // Synchronize to ensure Halide work completes before PyTorch accesses outputs
            // printf("[fusedssim_mps] Synchronizing after Halide kernel\n");
            // fflush(stdout);
            torch::mps::synchronize();
            // printf("[fusedssim_mps] Synchronization complete\n");

            // Clean up Halide buffers
            // printf("[fusedssim_mps] Cleaning up buffers\n");
            cleanup_halide_buffer(&buf1, mtl_buf1);
            cleanup_halide_buffer(&buf2, mtl_buf2);
            cleanup_halide_buffer(&ssim_buf, mtl_buf_ssim);
            cleanup_halide_buffer(&dm_dmu1_buf, mtl_buf_dmu1);
            cleanup_halide_buffer(&dm_dsigma1_sq_buf, mtl_buf_dsigma1);
            cleanup_halide_buffer(&dm_dsigma12_buf, mtl_buf_dsigma12);
            // printf("[fusedssim_mps] All buffers cleaned up\n");

            // Halide manages its own Metal context lifecycle
            // printf("[fusedssim_mps] Halide manages its own Metal context\n");

        } catch (...) {
            // Cleanup on error
            // printf("[fusedssim_mps] *** ERROR CAUGHT - Starting cleanup ***\n");
            // fflush(stdout);
            if (mtl_buf1) cleanup_halide_buffer(&buf1, mtl_buf1);
            if (mtl_buf2) cleanup_halide_buffer(&buf2, mtl_buf2);
            if (mtl_buf_ssim) cleanup_halide_buffer(&ssim_buf, mtl_buf_ssim);
            if (mtl_buf_dmu1) cleanup_halide_buffer(&dm_dmu1_buf, mtl_buf_dmu1);
            if (mtl_buf_dsigma1) cleanup_halide_buffer(&dm_dsigma1_sq_buf, mtl_buf_dsigma1);
            if (mtl_buf_dsigma12) cleanup_halide_buffer(&dm_dsigma12_buf, mtl_buf_dsigma12);
            // printf("[fusedssim_mps] *** Error cleanup complete, rethrowing ***\n");
            // fflush(stdout);
            throw;
        }

        // Return gradients only if training
        if (!train) {
            // printf("[fusedssim_mps] Not training, emptying gradient tensors\n");
            dm_dmu1_out = torch::empty({0}, img1.options());
            dm_dsigma1_sq_out = torch::empty({0}, img1.options());
            dm_dsigma12_out = torch::empty({0}, img1.options());
        }

        // printf("[fusedssim_mps] ========== Exiting fusedssim_mps successfully ==========\n\n");
        return std::make_tuple(ssim_out, dm_dmu1_out, dm_dsigma1_sq_out, dm_dsigma12_out);
    }
}

PYBIND11_MODULE(fused_ssim_halide_mps, m) {
    m.doc() = "Fused SSIM implementation using Halide + PyTorch MPS (zero-copy)";
    m.def("fusedssim", &fusedssim_mps, "Fused SSIM forward pass (MPS tensors, zero-copy)");
}
