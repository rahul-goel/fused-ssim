#include "binders.hpp"

namespace nb = nanobind;

NB_MODULE(gsplat_sycl_kernels, m) {
    m.def("fusedssim_forward", &fusedssim_forward);
    m.def("fusedssim_backward", &fusedssim_backward);
}