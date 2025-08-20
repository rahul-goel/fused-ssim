# setup.py

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

# A CMakeExtension needs a sourcedir instead of a file list.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        cfg = "Debug" if self.debug else "Release"
        
        # Configure CMake
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        
        # Build arguments
        build_args = ["--config", cfg]

        # Run CMake
        subprocess.run(["cmake", ext.sourcedir] + cmake_args, check=True)
        subprocess.run(["cmake", "--build", "."] + build_args, check=True)

setup(
    ext_modules=[CMakeExtension("fusedssim._C")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)