// PyTorch extension wrapper for Halide-generated SSIM
// CPU-only implementation using Halide buffers

#include <torch/extension.h>
#include "ssim_halide.h"
#include "HalideBuffer.h"

#include <tuple>
#include <stdexcept>

// Helper to create a Halide buffer wrapping PyTorch CPU tensor data
// PyTorch layout: NCHW (batch, channels, height, width)
// Halide layout: WHC (width, height, channels)
template<typename T>
Halide::Runtime::Buffer<T> tensor_to_halide_buffer(torch::Tensor& tensor) {
    // Ensure tensor is contiguous and on CPU
    tensor = tensor.contiguous();
    if (tensor.device().type() != torch::kCPU) {
        throw std::runtime_error("Only CPU tensors are supported. Use .cpu() to copy tensors to CPU first.");
    }

    // Get dimensions (NCHW)
    int batch = tensor.size(0);
    int channels = tensor.size(1);
    int height = tensor.size(2);
    int width = tensor.size(3);

    // For now, only support batch size 1
    if (batch != 1) {
        throw std::runtime_error("Halide SSIM currently only supports batch size 1");
    }

    // Create Halide buffer with WHC dimensions
    // The data pointer points to the tensor's underlying storage
    // We need to set up strides to match PyTorch's NCHW layout
    //
    // PyTorch NCHW strides: [C*H*W, H*W, W, 1]
    // For a single batch item starting at channel 0:
    // - x (width) has stride 1
    // - y (height) has stride W
    // - c (channel) has stride H*W
    //
    // Halide buffer dimensions are specified in order [dim0, dim1, dim2, ...]
    // with dim0 being the innermost (fastest-varying)
    // We want: dim0=x (stride 1), dim1=y (stride W), dim2=c (stride H*W)

    halide_dimension_t dims[3] = {
        {0, width, 1},           // x: extent=width, stride=1
        {0, height, width},      // y: extent=height, stride=width
        {0, channels, height * width}  // c: extent=channels, stride=H*W
    };

    Halide::Runtime::Buffer<T> buf(
        tensor.data_ptr<T>(),
        3,  // number of dimensions
        dims
    );

    return buf;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedssim(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    bool train
) {
    // Validate inputs
    if (img1.dim() != 4 || img2.dim() != 4) {
        throw std::runtime_error("Input tensors must be 4D (NCHW)");
    }
    if (img1.dtype() != torch::kFloat32 || img2.dtype() != torch::kFloat32) {
        throw std::runtime_error("Input tensors must be float32");
    }
    if (img1.sizes() != img2.sizes()) {
        throw std::runtime_error("Input tensors must have the same shape");
    }
    if (img1.device().type() != torch::kCPU || img2.device().type() != torch::kCPU) {
        throw std::runtime_error("Only CPU tensors are supported. Use .cpu() to copy tensors to CPU first.");
    }

    // Allocate output tensors on CPU
    torch::Tensor ssim_out = torch::empty_like(img1);
    torch::Tensor dm_dmu1_out = train ? torch::empty_like(img1) : torch::empty({0}, torch::kFloat32);
    torch::Tensor dm_dsigma1_sq_out = train ? torch::empty_like(img1) : torch::empty({0}, torch::kFloat32);
    torch::Tensor dm_dsigma12_out = train ? torch::empty_like(img1) : torch::empty({0}, torch::kFloat32);

    // Wrap PyTorch tensors as Halide buffers
    auto buf1 = tensor_to_halide_buffer<float>(img1);
    auto buf2 = tensor_to_halide_buffer<float>(img2);
    auto ssim_buf = tensor_to_halide_buffer<float>(ssim_out);

    // For training, wrap gradient output buffers
    Halide::Runtime::Buffer<float> dm_dmu1_buf, dm_dsigma1_sq_buf, dm_dsigma12_buf;
    if (train) {
        dm_dmu1_buf = tensor_to_halide_buffer<float>(dm_dmu1_out);
        dm_dsigma1_sq_buf = tensor_to_halide_buffer<float>(dm_dsigma1_sq_out);
        dm_dsigma12_buf = tensor_to_halide_buffer<float>(dm_dsigma12_out);
    } else {
        // Create empty buffers for non-training mode
        dm_dmu1_buf = Halide::Runtime::Buffer<float>();
        dm_dsigma1_sq_buf = Halide::Runtime::Buffer<float>();
        dm_dsigma12_buf = Halide::Runtime::Buffer<float>();
    }

    // Call the Halide-generated function
    int err = fused_ssim_forward(
        buf1,
        buf2,
        C1,
        C2,
        ssim_buf,
        dm_dmu1_buf,
        dm_dsigma1_sq_buf,
        dm_dsigma12_buf
    );

    if (err != 0) {
        throw std::runtime_error("Halide fused_ssim_forward failed with error code: " + std::to_string(err));
    }

    return std::make_tuple(ssim_out, dm_dmu1_out, dm_dsigma1_sq_out, dm_dsigma12_out);
}

torch::Tensor
fusedssim_backward(
    float C1,
    float C2,
    torch::Tensor& img1,
    torch::Tensor& img2,
    torch::Tensor& dL_dmap,
    torch::Tensor& dm_dmu1,
    torch::Tensor& dm_dsigma1_sq,
    torch::Tensor& dm_dsigma12
) {
    // Validate inputs are on CPU
    if (img1.device().type() != torch::kCPU) {
        throw std::runtime_error("Only CPU tensors are supported. Use .cpu() to copy tensors to CPU first.");
    }

    // TODO: Implement backward pass
    // For now, return a zero gradient tensor
    return torch::zeros_like(img1);
}

PYBIND11_MODULE(fused_ssim_halide, m) {
    m.doc() = "Fused SSIM implementation using Halide (CPU-only)";
    m.def("fusedssim", &fusedssim, "Fused SSIM forward pass (CPU tensors only)");
    m.def("fusedssim_backward", &fusedssim_backward, "Fused SSIM backward pass (CPU tensors only)");
}
