#include "bindings.hpp"

NB_MODULE(fusedssim_sycl_kernels , m) {
    m.def("fusedssim_forward_kernel_call", &fusedssim_forward_kernel_call);
    m.def("fusedssim_backward_kernel_call", &fusedssim_backward_kernel_call);
}