#include <torch/extension.h>
#include "ssim3d.h"
#include "ssim.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim3d", &fusedssim3d);
  m.def("fusedssim_backward3d", &fusedssim_backward3d);
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}
