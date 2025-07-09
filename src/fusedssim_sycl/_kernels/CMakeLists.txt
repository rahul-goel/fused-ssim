nanobind_add_module(fusedssim_sycl_kernels MODULE

${CMAKE_CURRENT_SOURCE_DIR}/src/ext.cpp

${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_backward_kernel_call.cpp
${CMAKE_CURRENT_SOURCE_DIR}/src/fusedssim_forward_kernel_call.cpp
)

target_include_directories( fusedssim_sycl_kernels SYSTEM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
)

target_compile_options(fusedssim_sycl_kernels PRIVATE -fsycl)
target_link_options(fusedssim_sycl_kernels PRIVATE -fsycl -fsycl-targets=${SYCL_AOT_TARGETS})
target_link_libraries(fusedssim_sycl_kernels PRIVATE IntelSYCL::SYCL_CXX torch)
target_compile_features(fusedssim_sycl_kernels PUBLIC cxx_std_17)
# Fix for icx: error: '-MP' is not supported with offloading enabled
if (WIN32 AND NOT UNIX)
    get_target_property(CURRENT_OPTIONS fusedssim_sycl_kernels COMPILE_OPTIONS)
    string(REPLACE "/MP" "" MODIFIED_OPTIONS "${CURRENT_OPTIONS}")
    set_target_properties(fusedssim_sycl_kernels PROPERTIES COMPILE_OPTIONS "${MODIFIED_OPTIONS}")
endif ()
if (UNIX AND NOT APPLE)
   # Find libtorch_xpu and libsycl at runtime in Python environment
   set_target_properties(fusedssim_sycl_kernels PROPERTIES INSTALL_RPATH 
           "$ORIGIN/../../torch/lib/;$ORIGIN/../../../../")
endif()
install(TARGETS fusedssim_sycl_kernels
    LIBRARY DESTINATION "fusedssim_sycl/_kernels"
)
