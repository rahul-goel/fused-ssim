#include <torch/extension.h>
#include "ssim.h"

// Only include 3D SSIM for CUDA builds
#ifdef FUSED_SSIM_CUDA
#include "ssim3d.h"
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // 2D SSIM (available on all backends)
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);

#ifdef FUSED_SSIM_CUDA
  // 2D mean-only training path and fused L1 + SSIM (CUDA only)
  m.def("fusedssim_sum", &fusedssim_sum);
  m.def("fusedssim_backward_mean", &fusedssim_backward_mean);

  // 3D SSIM (CUDA only for now)
  m.def("fusedssim3d", &fusedssim3d);
  m.def("fusedssim_backward3d", &fusedssim_backward3d);
#endif
}
