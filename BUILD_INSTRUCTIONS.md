# Phase 1 Build Instructions

## Prerequisites Check

### 1. Verify CUDA Installation
```powershell
nvcc --version
# Should show: CUDA compilation tools, release 12.6
```

### 2. Verify Visual Studio C++ Tools
```powershell
# Check if cl.exe (MSVC compiler) is available
where cl
# If not found, install "Desktop development with C++" from Visual Studio Installer
```

### 3. Install CMake
```powershell
# Check if CMake is installed
cmake --version

# If not installed:
# Download from https://cmake.org/download/
# Or use chocolatey:
choco install cmake
```

### 4. Install Python Dependencies
```powershell
# Activate your virtual environment first
D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/Activate.ps1

# Install required packages
pip install pybind11 pytest numpy
```

---

## Build Method 1: Using CMake Directly (Recommended)

### Step 1: Clone/Download pybind11
```powershell
cd D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43

# Option A: Git clone (if you have git)
git clone https://github.com/pybind/pybind11.git

# Option B: Download and extract manually
# Download from: https://github.com/pybind/pybind11/archive/refs/heads/master.zip
# Extract to project root
```

### Step 2: Configure with CMake
```powershell
# Create build directory
mkdir build
cd build

# Configure (this generates Visual Studio solution)
cmake .. -G "Visual Studio 16 2019" -A x64 `
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" `
    -DPYTHON_EXECUTABLE="D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/python.exe"

# If you have Visual Studio 2022:
cmake .. -G "Visual Studio 17 2022" -A x64 `
    -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe" `
    -DPYTHON_EXECUTABLE="D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/python.exe"
```

### Step 3: Build
```powershell
# Build the project
cmake --build . --config Release

# This will create: planar_cuda.pyd (Python module)
```

### Step 4: Copy Module to Project Root
```powershell
# Copy the compiled module to where Python can find it
copy Release\planar_cuda.pyd ..\planar_cuda.pyd

# Or add to Python path in your script
```

---

## Build Method 2: Using setup.py (Alternative)

```powershell
# From project root
python setup.py build_ext --inplace

# This will:
# 1. Create build/ directory
# 2. Run CMake automatically
# 3. Place planar_cuda.pyd in project root
```

---

## Testing the Build

### Test 1: Import Check
```powershell
python -c "import planar_cuda; print(planar_cuda.__version__)"
# Expected output: 0.1.0-phase1
```

### Test 2: Run Test Suite
```powershell
pytest tests/cuda_tests/test_phase1_pipeline.py -v
```

Expected output:
```
test_phase1_pipeline.py::test_module_import PASSED
test_phase1_pipeline.py::test_simple_addition PASSED
test_phase1_pipeline.py::test_large_vectors PASSED
test_phase1_pipeline.py::test_error_handling PASSED
test_phase1_pipeline.py::test_empty_vectors PASSED

✅ All tests passed!
```

---

## Troubleshooting

### Error: "CUDA_ARCHITECTURES is empty"
**Solution**: Edit `CMakeLists.txt`, set your GPU's compute capability:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4060
# 86 = RTX 3090
# 75 = RTX 2080
# 61 = GTX 1080
```

### Error: "Cannot open include file 'pybind11/pybind11.h'"
**Solution**: Make sure pybind11 is in project root or installed:
```powershell
pip install pybind11[global]
```

Or use system-wide pybind11 in CMakeLists.txt:
```cmake
find_package(pybind11 REQUIRED)  # Instead of add_subdirectory
```

### Error: "LINK : fatal error LNK1104: cannot open file 'python310.lib'"
**Solution**: Check Python paths:
```powershell
# Verify Python executable
where python

# Update CMake command with correct path:
-DPYTHON_EXECUTABLE="C:/path/to/your/python.exe"
```

### Error: Module imports but crashes on function call
**Solution**: CUDA DLL path issue. Add to your Python code:
```python
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda
```

---

## Success Criteria

✅ **Phase 1 Complete** when all these work:
1. `nvcc --version` shows CUDA 12.6
2. `cmake --version` shows CMake 3.18+
3. Build completes without errors
4. `import planar_cuda` works in Python
5. All 5 tests in `test_phase1_pipeline.py` pass
6. Vector addition returns correct results

---

## Next Steps After Phase 1

Once Phase 1 tests pass, you're ready for:
- **Phase 2**: Port geometry kernels (`cross_product`, `segments_intersect`)
- **Phase 3**: GPU memory management (`CudaGraph` class)
- **Phase 4**: Full SA solver on GPU

---

## Quick Start Script

Save as `build_and_test.ps1`:
```powershell
# Build and test Phase 1
$ErrorActionPreference = "Stop"

Write-Host "🔧 Building CUDA module..." -ForegroundColor Cyan
mkdir -Force build | Out-Null
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ..

Write-Host "📋 Copying module..." -ForegroundColor Cyan
copy build\Release\planar_cuda.pyd . -Force

Write-Host "🧪 Running tests..." -ForegroundColor Cyan
pytest tests/cuda_tests/test_phase1_pipeline.py -v

Write-Host "✅ Phase 1 Complete!" -ForegroundColor Green
```

Run it:
```powershell
.\build_and_test.ps1
```
