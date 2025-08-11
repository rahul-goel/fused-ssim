from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


# Default fallback architectures
fallback_archs = [
    "-gencode=arch=compute_75,code=sm_75",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_89,code=sm_89",
]

nvcc_args = [
    "-O3",
    "--maxrregcount=32",
    "--use_fast_math",
]

detected_arch = None

# Check for CUDA_ARCHITECTURES environment variable first
cuda_archs_env = os.environ.get('CUDA_ARCHITECTURES')
if cuda_archs_env:
    try:
        archs = [arch.strip() for arch in cuda_archs_env.split(';')]
        env_msg = f"Using CUDA architectures from environment: {archs}"
        print(env_msg)
        print(env_msg, file=sys.stderr, flush=True)

        for arch in archs:
            nvcc_args.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        detected_arch = f"env:{','.join(archs)}"
    except Exception as e:
        error_msg = f"Failed to parse CUDA_ARCHITECTURES environment variable: {e}. Trying device detection."
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        cuda_archs_env = None  # Reset to try device detection

if not cuda_archs_env and torch.cuda.is_available():
    try:
        device = torch.cuda.current_device()
        compute_capability = torch.cuda.get_device_capability(device)
        arch = f"sm_{compute_capability[0]}{compute_capability[1]}"

        # Print to multiple outputs
        arch_msg = f"Detected GPU architecture: {arch}"
        print(arch_msg)
        print(arch_msg, file=sys.stderr, flush=True)

        nvcc_args.append(f"-arch={arch}")
        detected_arch = arch
    except Exception as e:
        error_msg = f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures."
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)
        nvcc_args.extend(fallback_archs)
elif not cuda_archs_env:
    cuda_msg = "CUDA not available. Falling back to multiple architectures."
    print(cuda_msg)
    print(cuda_msg, file=sys.stderr, flush=True)
    nvcc_args.extend(fallback_archs)

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_ssim",
    packages=['fused_ssim'],
    ext_modules=[
        CUDAExtension(
            name="fused_ssim_cuda",
            sources=[
                "ssim.cu",
                "ext.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)