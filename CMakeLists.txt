cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_C_COMPILER icx)
set(CMAKE_CXX_COMPILER icpx)

project(gsplat_sycl)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_compile_options(-fsycl -fsycl-targets=spir64,intel_gpu_acm_g10) 
add_link_options(-fsycl -fsycl-targets=spir64,intel_gpu_acm_g10)

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

find_package(IntelSYCL REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE Torch_DIR_From_Python 
  RESULT_VARIABLE _torch_path_result    
  ERROR_QUIET                           
)
if(NOT _torch_path_result EQUAL 0)
  message(WARNING "Failed to get Torch CMake path from Python. "
                  "Make sure PyTorch is installed in the Python environment: ${Python_EXECUTABLE}")
  set(Torch_DIR_From_Python "")
endif()

if(Torch_DIR_From_Python AND IS_DIRECTORY "${Torch_DIR_From_Python}")
  set(Torch_DIR ${Torch_DIR_From_Python})
  message(STATUS "Found Torch CMake directory via Python: ${Torch_DIR}")
  find_package(Torch REQUIRED HINTS ${Torch_DIR_From_Python})
else()
  message(FATAL_ERROR "Could not find Torch via Python introspection. "
                      "Please ensure PyTorch is installed or set CMAKE_PREFIX_PATH/Torch_DIR manually.")
endif()

add_subdirectory(src/fusedssim_sycl/_kernels)