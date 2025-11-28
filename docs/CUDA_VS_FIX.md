# CUDA Integration Fix for Visual Studio 2022

## Problem
CMake cannot find CUDA toolset because Visual Studio 2022 needs CUDA integration installed.

## Solution Options

### Option 1: Install CUDA Integration for VS2022 (Recommended)

1. Run **CUDA Installer** again:
   ```powershell
   # Download from: https://developer.nvidia.com/cuda-downloads
   # Or run the installer you already have
   ```

2. During installation, make sure to select:
   - [x] **Visual Studio Integration**
   - [x] **Development** (Compiler, Libraries, Tools)

3. The installer will add CUDA support to Visual Studio 2022

4. After installation, retry the build

### Option 2: Use NMake instead of Visual Studio

Use this manual build command:

```powershell
# Clean
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
mkdir build
cd build

# Configure with NMake
cmake .. -G "NMake Makefiles" `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"

# Build
nmake

# Copy module
cd ..
copy build\planar_cuda.pyd .
```

### Option 3: Use setuptools with nvcc directly

Create `setup_direct.py`:

```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os

class CUDAExtension(Extension):
    pass

class custom_build_ext(build_ext):
    def build_extensions(self):
        # Compile CUDA
        nvcc = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
        subprocess.check_call([
            nvcc, "-c",
            "src/cuda_utils/vector_ops.cu",
            "-o", "build/vector_ops.obj",
            "-arch=sm_89",
            "--compiler-options", "/MD",
            "-std=c++17"
        ])
        
        # Add compiled object to extension
        self.extensions[0].extra_objects = ["build/vector_ops.obj"]
        
        # Build normally
        build_ext.build_extensions(self)

setup(
    name='planar_cuda',
    ext_modules=[
        CUDAExtension(
            'planar_cuda',
            sources=['src/cpp_binding/binding.cpp'],
            include_dirs=[
                r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\include',
                'src'
            ],
            library_dirs=[r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64'],
            libraries=['cudart'],
        )
    ],
    cmdclass={'build_ext': custom_build_ext},
)
```

Then run:
```powershell
python setup_direct.py build_ext --inplace
```

## Quick Test Which Option Works

Try Option 2 first (easiest):

```powershell
# Activate venv
D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/Activate.ps1

# Clean
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
mkdir build
cd build

# Find nmake
where nmake
# If not found, run: "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# Configure with NMake  
cmake .. -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release

# Build
nmake

# Test
cd ..
copy build\planar_cuda.pyd .
python -c "import os; os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'); import planar_cuda; print(planar_cuda.__version__)"
```

## Recommended Action

**Try Option 2 (NMake)** - it bypasses Visual Studio project generation and works directly with make files.
