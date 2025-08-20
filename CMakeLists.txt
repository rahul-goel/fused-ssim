cmake_minimum_required(VERSION 3.23)
project(fusedssim)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python: ${Python_EXECUTABLE}")

find_package(pybind11 REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE Torch_DIR_From_Python
    RESULT_VARIABLE _torch_path_result
    ERROR_QUIET
)

if(NOT _torch_path_result EQUAL 0)
    message(FATAL_ERROR "Failed to get Torch CMake path from Python. Is PyTorch installed?")
endif()

message(STATUS "Found Torch CMake directory via Python: ${Torch_DIR_From_Python}")
find_package(Torch REQUIRED HINTS ${Torch_DIR_From_Python})

set(BACKEND "NONE" CACHE INTERNAL "The selected compute backend")
set(KERNEL_SOURCES "" CACHE INTERNAL "Source files for the selected backend")

find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA toolkit found. Compiling for NVIDIA backend.")
    set(BACKEND "CUDA")
    list(APPEND KERNEL_SOURCES "fusedssim/cpp/src/cuda/ssim.cu")
    enable_language(CUDA)
else()
    message(STATUS "CUDA not found. Checking for Intel SYCL...")
    find_package(IntelSYCL REQUIRED)
    if(SYCL_FOUND)
        message(STATUS "Intel SYCL compiler found. Compiling for SYCL backend.")
        set(BACKEND "SYCL")
        list(APPEND KERNEL_SOURCES
            "fusedssim/cpp/src/xpu/fusedssim.cpp"
            "fusedssim/cpp/src/xpu/fusedssim_backward.cpp"
        )
        
        if (NOT SYCL_AOT_TARGETS)
            set (SYCL_AOT_TARGETS "spir64" CACHE STRING "SYCL target for AOT compilation.")
        endif()
        message(STATUS "Using SYCL AOT targets: ${SYCL_AOT_TARGETS}")

    endif()
endif()

if(BACKEND STREQUAL "NONE")
    message(FATAL_ERROR "No supported backend found! Neither CUDA nor Intel SYCL toolchain was detected.")
endif()

pybind11_add_module(_C SHARED fusedssim/cpp/src/ext.cpp ${KERNEL_SOURCES})

if(BACKEND STREQUAL "CUDA")
    target_compile_definitions(_C PRIVATE USE_CUDA)
elseif(BACKEND STREQUAL "SYCL")
    target_compile_definitions(_C PRIVATE USE_SYCL)
    target_compile_options(_C PRIVATE -fsycl-targets=${SYCL_AOT_TARGETS})
endif()

target_include_directories(_C PRIVATE fusedssim/cpp/include)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

if(BACKEND STREQUAL "SYCL")
    target_link_libraries(_C PRIVATE IntelSYCL::SYCL_CXX)
endif()

set_property(TARGET _C PROPERTY CXX_STANDARD 17)
if(MSVC)
    target_compile_options(_C PRIVATE /W4)
endif()

install(TARGETS _C LIBRARY DESTINATION ${PROJECT_NAME})