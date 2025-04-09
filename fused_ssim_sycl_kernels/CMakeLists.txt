nanobind_add_module(gsplat_sycl_kernels 

${CMAKE_CURRENT_SOURCE_DIR}/src/ext.cpp

${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_backward.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_forward.cpp
)

target_include_directories( gsplat_sycl_kernels SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    ${CMAKE_CURRENT_SOURCE_DIR}/external/glm 
)

target_link_libraries(gsplat_sycl_kernels PRIVATE IntelSYCL::SYCL_CXX torch)
target_compile_features(gsplat_sycl_kernels PUBLIC cxx_std_17)
