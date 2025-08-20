# setup.py

import os
import subprocess
import sys
from pathlib import Path
import shutil  

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

has_cuda = False
try:
    import torch
    has_cuda = torch.cuda.is_available()
except ImportError:
    pass

has_xpu = False
if not has_cuda: 
    try:
        import torch
        has_xpu = torch.xpu.is_available()
    except (ImportError, AttributeError):
        pass 

has_sycl_compiler = False
if os.system('icpx --version > /dev/null 2>&1') == 0:
    has_sycl_compiler = True
elif os.system('dpcpp --version > /dev/null 2>&1') == 0:
    has_sycl_compiler = True

BUILD_SYCL = has_xpu and has_sycl_compiler

class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Debug" if self.debug else "Release"
        
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        
        if BUILD_SYCL:
            print("--- Found 'icx' compiler, configuring for SYCL/XPU backend. ---")
            cmake_args.extend([
                "-D CMAKE_C_COMPILER=icx",
                "-D CMAKE_CXX_COMPILER=icx",
            ])
        
        build_args = ["--config", cfg]

        build_dir = Path(self.build_temp)
        build_dir.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir] + cmake_args, check=True, cwd=build_dir)
        subprocess.run(["cmake", "--build", "."] + build_args, check=True, cwd=build_dir)

setup(
    ext_modules=[CMakeExtension("fusedssim._C", sourcedir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)