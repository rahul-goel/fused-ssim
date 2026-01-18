#!/bin/bash
# Halide SSIM AOT Generator Script
# Usage: ./gen.sh [target]
# target: metal, cuda, opencl, or host (default)

set -e  # Exit on error

TARGET=${1:-host}

# Halide paths from conda environment
export HALIDE_PATH=$CONDA_PREFIX/lib/python3.13/site-packages/halide
export HALIDE_H_PATH=$HALIDE_PATH/include
export LIB_HALIDE_PATH=$HALIDE_PATH/lib

echo "==================================="
echo "Halide SSIM AOT Generator"
echo "==================================="
echo "Target: $TARGET"
echo "HALIDE_PATH: $HALIDE_PATH"
echo ""

# Step 1: Compile the generator (needs Halide at compile time)
echo "Step 1: Compiling generator..."
g++ fused_ssim.cpp -o fused_ssim_gen \
    -I $HALIDE_H_PATH \
    -L $LIB_HALIDE_PATH \
    -lHalide \
    -lpthread \
    -std=c++17 \
    -O2

echo "Generator compiled: ./fused_ssim_gen"
echo ""

# Step 2: Run the generator to produce ssim_halide.a and ssim_halide.h
echo "Step 2: Running generator for target '$TARGET'..."
export DYLD_LIBRARY_PATH=$LIB_HALIDE_PATH:$DYLD_LIBRARY_PATH
./fused_ssim_gen $TARGET

echo ""
echo "==================================="
echo "Generated files:"
ls -lh ssim_halide.* 2>/dev/null || echo "  (no files generated yet)"
echo "==================================="
echo ""
echo "To build the PyTorch extension (CPU-only):"
echo "  ./build.sh"
