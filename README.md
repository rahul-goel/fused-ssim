# Fully Fused Differentiable SSIM

## XPU Installation Instructions
- You must have pytorch working on XPU installed in you Python 3.X environment. This project has currently been tested with:
  - pytorch for xpu on Intel GPUs
  - install `nanobind` for packaging
  - install `pytorch_mssim` for comparison

### Build
```
git clone git@github.com:isl-org/fused-ssim.git
```
#### Build on Linux

* Setup environment for OneAPI:
```
source /opt/intel/oneapi/setvars.sh
```
The OneAPI version must match the OneAPI version used in your PyTorch XPU. (e.g. both can be 2025.0.*)

* Install:
```
pip install --no-build-isolation .
```
The `--no-build-isolation` flag is necessary for fused-ssim to find and link to PyTorch libraries.

Alternately, you can build a wheel for distribution with:
```
python -m build --no-isolation --wheel
```

#### Build on Windows
* Setup environment with MSBuild tools and OneAPI in the command prompt:

```
cmd /k "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
powershell
cmd /k "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
powershell
```
The OneAPI version must match the OneAPI version used in your PyTorch XPU. (e.g. both can be 2025.0.*)

Now install or build a wheel as in the Linux instructions above.

### Test
```
python -m tests.test
```

## Benchmarking

Same script as provided by authors (tests/test.py)

Initial numbers
```    
Reference Time (Forward): 79.56446647644043 ms
Reference Time (Backward): 123.47569227218628 ms
pytorch_mssim Time (Forward): 89.01119470596313 ms
pytorch_mssim Time (Backward): 88.22749614715576 ms
fused-ssim Time (Forward): 21.227867603302002 ms
fused-ssim Time (Backward): 24.689362049102783 ms
fused-ssim Time (Inference): 14.284882545471191 ms
```

## TODO

Add all references and update readme
