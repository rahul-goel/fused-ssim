from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
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
detected_arch = None

compiler_args = {"cxx": ["-O3"], "nvcc": ["-O3"]}
link_args = []
if torch.cuda.is_available():
    if torch.version.hip:
        hip_msg = "Detected AMD GPU with ROCm/HIP"
        print(hip_msg)
        print(hip_msg, file=sys.stderr, flush=True)
        compiler_args["nvcc"].append("-ffast-math")
    else:
        compiler_args["nvcc"].extend(("--maxrregcount=32", "--use_fast_math"))
        try:
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            arch = f"sm_{compute_capability[0]}{compute_capability[1]}"

            # Print to multiple outputs
            arch_msg = f"Detected GPU architecture: {arch}"
            print(arch_msg)
            print(arch_msg, file=sys.stderr, flush=True)

            compiler_args["nvcc"].append(f"-arch={arch}")
            detected_arch = arch
        except Exception as e:
            error_msg = f"Failed to detect GPU architecture: {e}. Falling back to multiple architectures."
            print(error_msg)
            print(error_msg, file=sys.stderr, flush=True)
            compiler_args["nvcc"].extend(fallback_archs)

elif torch.mps.is_available():
    extension_type = CppExtension

    extension_file = "ssim.mm"
    build_name = "fused_ssim_mps"

    compiler_args["cxx"] += ["-std=c++17", "-ObjC++", "-Wno-unused-parameter"]
    link_args += ["-framework", "Metal", "-framework", "Foundation"]

    detected_arch = "Apple Silicon (MPS)"

else:
    cuda_msg = "CUDA not available. Falling back to multiple architectures."
    print(cuda_msg)
    print(cuda_msg, file=sys.stderr, flush=True)
    compiler_args["nvcc"].extend(fallback_archs)

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
final_msg = "Setup completed. NVCC args: {}. CXX args: {}. Link args: {}.".format(compiler_args["nvcc"],compiler_args["cxx"], link_args)
print(final_msg)
