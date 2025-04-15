nanobind_add_module(fusedssim_sycl_kernels MODULE

${CMAKE_CURRENT_SOURCE_DIR}/src/ext.cpp

${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_backward_kernel_call.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_forward_kernel_call.cpp
)

target_include_directories( fusedssim_sycl_kernels SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_link_libraries(fusedssim_sycl_kernels PRIVATE IntelSYCL::SYCL_CXX torch)
target_compile_features(fusedssim_sycl_kernels PUBLIC cxx_std_17)
install(TARGETS fusedssim_sycl_kernels
    LIBRARY DESTINATION "fusedssim_sycl/_kernels"
)
