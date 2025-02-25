from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

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
                "nvcc": [
                    "-O3",
                    "--maxrregcount=32",
                    "--use_fast_math",
                    "-arch=sm_89",  # Adjust for your GPU
                ]
            }
            )
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
