# Fully Fused Differentiable SSIM

## XPU Installation Instructions
- You must have pytorch working on XPU installed in you Python 3.X environment. This project has currently been tested with:
  - pytorch2.6 for xpu on intel A770 
  - install nanobing for packaging
  - install pytorch_mssim for comparison

### Build
```
git clone git@github.com:isl-org/fused-ssim.git
pip install -e .
```

**Note**: If you get the following CMake error even if the virtual environment has pytorch
```
CMake Error at CMakeLists.txt:48 (message):
        Could not find Torch via Python introspection.  Please ensure PyTorch is
        installed or set CMAKE_PREFIX_PATH/Torch_DIR manually.
``` 
use the following command
```
pip install --no-build-isolation -e .
```

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
