#include <torch/extension.h>
#include "fusedssim/ssim.hpp"

PYBIND11_MODULE(_C, m) {
  m.def("fusedssim", &fusedssim);
  m.def("fusedssim_backward", &fusedssim_backward);
}