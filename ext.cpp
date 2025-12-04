#include <torch/extension.h>
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
  m.def("fusedl1ssim_loss", &fusedl1ssim_loss);
  m.def("fusedl1ssim_loss_backward", &fusedl1ssim_loss_backward);
}
