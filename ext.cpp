#include <torch/extension.h>
#include "ssim3d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fusedssim3D", &fusedssim3D);
  m.def("fusedssim_backward3D", &fusedssim_backward3D);
}
