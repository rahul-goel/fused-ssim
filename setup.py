from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


def log(msg):
    """Print to both stdout and stderr."""
    print(msg)
    print(msg, file=sys.stderr, flush=True)


def configure_cuda():
    """Configure CUDA/ROCm backend."""
    fallback_archs = [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_89,code=sm_89",
    ]

    log("Compiling for CUDA.")
    compiler_args = {"cxx": ["-O3"], "nvcc": ["-O3"]}

    if torch.version.hip:
        log("Detected AMD GPU with ROCm/HIP")
        compiler_args["nvcc"].append("-ffast-math")
        detected_arch = "AMD GPU (ROCm/HIP)"
    else:
        compiler_args["nvcc"].extend(("--maxrregcount=32", "--use_fast_math"))
        try:
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
            log(f"Detected GPU architecture: {arch}")
            compiler_args["nvcc"].append(f"-arch={arch}")
            detected_arch = arch
        except Exception as e:
            log(f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures.")
            compiler_args["nvcc"].extend(fallback_archs)
            detected_arch = "multiple architectures"

    return CUDAExtension, "ssim3d.cu", "fused_ssim3d_cuda", compiler_args, [], detected_arch

extension_type, extension_file, build_name, compiler_args, link_args, detected_arch = configure_cuda()

# Create a custom class that prints the architecture information
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # # For SYCL, override compiler to use icpx
        # if 'xpu' in build_name:
        #     self.compiler.compiler_so = ['icpx'] + self.compiler.compiler_so[1:]
        #     self.compiler.compiler_cxx = ['icpx'] + self.compiler.compiler_cxx[1:]
        #     self.compiler.linker_so = ['icpx'] + self.compiler.linker_so[1:]

        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "="*50)
        print(arch_info)
        print("="*50 + "\n")
        super().build_extensions()

setup(
    name="fused_ssim3d",
    packages=['fused_ssim3d'],
    ext_modules=[
        extension_type(
            name=build_name,
            sources=[
                extension_file,
                "ext.cpp"],
            extra_compile_args=compiler_args,
            extra_link_args=link_args
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

# Print again at the end of setup.py execution
if "nvcc" in compiler_args:
    final_msg = "Setup completed. NVCC args: {}. CXX args: {}. Link args: {}.".format(
        compiler_args["nvcc"], compiler_args["cxx"], link_args
    )
else:
    final_msg = "Setup completed. CXX args: {}. Link args: {}.".format(
        compiler_args["cxx"], link_args
    )
print(final_msg)
