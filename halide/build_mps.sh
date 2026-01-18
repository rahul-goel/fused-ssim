#!/bin/bash
# Build the PyTorch MPS extension from Halide-generated library
# Zero-copy integration using Metal buffers
# Usage: ./build_mps.sh

set -e  # Exit on error

# Halide paths
export HALIDE_PATH=$CONDA_PREFIX/lib/python3.13/site-packages/halide
export HALIDE_H_PATH=$HALIDE_PATH/include
export LIB_HALIDE_PATH=$HALIDE_PATH/lib

echo "==================================="
echo "Building PyTorch MPS Extension (Zero-Copy)"
echo "==================================="

# Check that ssim_halide.a exists and was built for Metal
if [ ! -f ssim_halide.a ]; then
    echo "Error: ssim_halide.a not found. Run './gen.sh metal' first."
    exit 1
fi
echo "Found ssim_halide.a"

# Get Python/PyTorch include paths
PYTHON_INCLUDES=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_INCLUDES=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
TORCH_LIBS=$(python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
PYBIND_INCLUDES=$(python3 -m pybind11 --includes)

echo "Python includes: $PYTHON_INCLUDES"
echo "Torch includes: $TORCH_INCLUDES"
echo "Torch libs: $TORCH_LIBS"
echo ""

# Compile the MPS extension (using .mm for Objective-C++)
echo "Compiling ext_halide_mps.mm..."
clang++ ext_halide_mps.mm ssim_halide.a \
    -shared -fPIC \
    -I $HALIDE_H_PATH \
    -I $PYTHON_INCLUDES \
    $TORCH_INCLUDES \
    $PYBIND_INCLUDES \
    $TORCH_LIBS \
    -ltorch -ltorch_cpu -lc10 \
    -framework Metal \
    -framework Foundation \
    -std=c++17 \
    -O2 \
    -undefined dynamic_lookup \
    -fobjc-arc \
    -o fused_ssim_halide_mps$(python3-config --extension-suffix)

echo ""
echo "==================================="
echo "Build complete! (MPS Zero-Copy)"
echo "Extension: fused_ssim_halide_mps.cpython-*.so"
ls -la fused_ssim_halide_mps*.so
echo ""
echo "Usage: import fused_ssim_halide_mps"
echo "Note: Requires MPS tensors - zero-copy operation!"
echo "==================================="
