#include <torch/extension.h>
#include "ssim3d.h"
#include "ssim2d.h"
#include "ssim3d_by_2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim3d", &fusedssim3d);
  m.def("fusedssim_backward3d", &fusedssim_backward3d);
  m.def("fusedssim2d", &fusedssim2d);
  m.def("fusedssim_backward2d", &fusedssim_backward2d);
  m.def("fusedssim3d_by_2d", &fusedssim3d_by_2d);
  m.def("fusedssim_backward3d_by_2d", &fusedssim_backward3d_by_2d);
}
