# Python-CUDA Binding Guide

Complete guide for building and using the Python-CUDA interface using pybind11.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Building the Module](#building-the-module)
4. [Using the Module](#using-the-module)
5. [File Structure](#file-structure)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

---

## Overview

This project uses **pybind11** to create Python bindings for CUDA-accelerated C++ functions. The binding allows seamless data transfer between Python, C++, and GPU.

**Technology Stack:**
- **Python 3.11.4** - Host language
- **pybind11** - C++/Python binding library
- **CUDA 12.6.20** - GPU computation
- **MSVC 19.44** - C++ compiler (Visual Studio 2022)
- **nvcc** - NVIDIA CUDA compiler

---

## Architecture

### Data Flow

```
Python Code
    ↓ (pybind11)
C++ Wrapper
    ↓ (CUDA Runtime API)
GPU Kernel
    ↓ (cudaMemcpy)
Python Result
```

### Component Breakdown

#### 1. CUDA Kernel (`src/cuda_utils/vector_ops.cu`)

```cpp
// GPU code: runs on device
__global__ void add_vectors_kernel(const int* a, const int* b, int* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Host wrapper: manages memory and kernel launch
std::vector<int> add_vectors_gpu(const std::vector<int>& a, const std::vector<int>& b) {
    // 1. Validate input
    // 2. Allocate GPU memory
    // 3. Copy data to GPU
    // 4. Launch kernel
    // 5. Copy results back
    // 6. Free GPU memory
    return result;
}
```

#### 2. pybind11 Binding (`src/cpp_binding/binding.cpp`)

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // Automatic std::vector conversion

namespace py = pybind11;

// Forward declaration from .cu file
std::vector<int> add_vectors_gpu(const std::vector<int>& a, const std::vector<int>& b);

// Module definition
PYBIND11_MODULE(planar_cuda, m) {
    m.doc() = "LCN Solver - CUDA Accelerated Backend";
    
    // Expose function to Python
    m.def("add_vectors", &add_vectors_gpu, 
          "Add two integer vectors using CUDA",
          py::arg("a"), py::arg("b"));
    
    // Module metadata
    m.attr("__version__") = "0.1.0-phase1";
    m.attr("cuda_enabled") = true;
}
```

**Key Features:**
- `pybind11/stl.h` - Automatic conversion between `std::vector<int>` and Python `list`
- `py::arg()` - Named arguments in Python
- `m.attr()` - Module-level constants

---

## Building the Module

### Prerequisites

1. **CUDA Toolkit 12.6+**
   ```powershell
   # Verify installation
   nvcc --version
   # Expected: cuda_12.6.r12.6
   ```

2. **Visual Studio 2022 Community**
   - Must include "Desktop development with C++"
   - MSVC compiler v19.44+

3. **Python 3.9+**
   ```powershell
   python --version
   # Expected: Python 3.11.4
   ```

4. **pybind11**
   ```powershell
   pip install pybind11
   ```

### Build Process

#### Method 1: Automated Build (Recommended)

```powershell
# Navigate to project root
cd D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43

# Run build script
.\scripts\build_final.ps1
```

**What the script does:**
1. Initializes MSVC environment (vcvars64.bat)
2. Compiles CUDA kernel (`.cu` → `.obj`)
3. Compiles C++ binding (`.cpp` → `.obj`)
4. Links into Python module (`.pyd`)
5. Runs basic import test
6. Outputs to `build_artifacts/planar_cuda.pyd`

**Expected Output:**
```
========================================
  CUDA Build with MSVC Environment
========================================

Compiling CUDA kernel...
Compiling C++ binding...
Linking module...
Build successful!

Testing module...
Version: 0.1.0-phase1
CUDA enabled: True
Test: [1,2,3] + [4,5,6] = [5, 7, 9]
SUCCESS!

========================================
  BUILD SUCCESSFUL!
========================================
```

#### Method 2: Manual Build

```powershell
# 1. Initialize MSVC environment
cmd /c "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

# 2. Create build directory
mkdir build_artifacts -Force

# 3. Compile CUDA kernel
nvcc -c src/cuda_utils/vector_ops.cu `
    -o build_artifacts/vector_ops.obj `
    -arch=sm_89 `
    --compiler-options "/EHsc /MD" `
    -Isrc

# 4. Get pybind11 include path
$pybind11_include = python -c "import pybind11; print(pybind11.get_include())"

# 5. Compile C++ binding
nvcc -c src/cpp_binding/binding.cpp `
    -o build_artifacts/binding.obj `
    --compiler-options "/EHsc /MD" `
    -Isrc `
    -I"C:\Users\aloha\AppData\Local\Programs\Python\Python311\Include" `
    -I"$pybind11_include"

# 6. Link into Python module
nvcc --shared `
    build_artifacts/vector_ops.obj `
    build_artifacts/binding.obj `
    -o build_artifacts/planar_cuda.pyd `
    -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64" `
    -L"C:\Users\aloha\AppData\Local\Programs\Python\Python311\libs" `
    -lcudart -lpython311 `
    -Xlinker "/NODEFAULTLIB:MSVCRT" `
    -Xlinker "legacy_stdio_definitions.lib" `
    -Xlinker "ucrt.lib" `
    -Xlinker "vcruntime.lib"
```

### Build Flags Explained

| Flag | Purpose |
|------|---------|
| `-arch=sm_89` | Target RTX 4060 (compute capability 8.9) |
| `/EHsc` | Enable C++ exception handling |
| `/MD` | Link with multithreaded DLL runtime |
| `-Isrc` | Include source directory |
| `--shared` | Create shared library (.pyd) |
| `-lcudart` | Link CUDA runtime |
| `legacy_stdio_definitions.lib` | Fix C runtime symbol resolution |

---

## Using the Module

### Basic Usage

```python
import os
import sys

# Step 1: Add CUDA DLL directory (Windows only)
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')

# Step 2: Add build_artifacts to Python path
sys.path.insert(0, r'D:\D_backup\2025\tum\25W\hackthon\Hackathon-Nov-25-Heilbronn43\build_artifacts')

# Step 3: Import module
import planar_cuda

# Step 4: Use CUDA functions
result = planar_cuda.add_vectors([1, 2, 3], [4, 5, 6])
print(result)  # Output: [5, 7, 9]
```

### Module API

#### `add_vectors(a, b)`

Add two integer vectors using GPU acceleration.

**Parameters:**
- `a` (list[int]): First input vector
- `b` (list[int]): Second input vector

**Returns:**
- `list[int]`: Element-wise sum of `a` and `b`

**Raises:**
- `RuntimeError`: If vector sizes don't match
- `RuntimeError`: If CUDA kernel launch fails

**Examples:**

```python
# Simple addition
result = planar_cuda.add_vectors([1, 2, 3], [4, 5, 6])
# Returns: [5, 7, 9]

# Large vectors (10,000 elements)
a = list(range(10000))
b = list(range(10000, 20000))
result = planar_cuda.add_vectors(a, b)
# Returns: [10000, 10002, 10004, ..., 29998]

# Empty vectors
result = planar_cuda.add_vectors([], [])
# Returns: []

# Error: mismatched sizes
try:
    result = planar_cuda.add_vectors([1, 2], [1, 2, 3])
except RuntimeError as e:
    print(e)  # Vector sizes must match
```

#### Module Attributes

```python
print(planar_cuda.__version__)    # "0.1.0-phase1"
print(planar_cuda.cuda_enabled)   # True
print(planar_cuda.__doc__)        # "LCN Solver - CUDA Accelerated Backend"
```

### Integration with Tests

```python
# tests/cuda_tests/test_phase1_pipeline.py
import pytest
import sys
import os
from pathlib import Path

# Setup paths
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
project_root = Path(__file__).parent.parent.parent
build_artifacts = project_root / "build_artifacts"
sys.path.insert(0, str(build_artifacts))

def test_simple_addition():
    import planar_cuda
    result = planar_cuda.add_vectors([1, 2, 3], [4, 5, 6])
    assert result == [5, 7, 9]
```

**Run tests:**
```powershell
pytest tests/cuda_tests/test_phase1_pipeline.py -v
```

---

## File Structure

```
project_root/
├── src/
│   ├── cuda_utils/
│   │   └── vector_ops.cu          # CUDA kernel implementation
│   └── cpp_binding/
│       └── binding.cpp             # pybind11 binding layer
├── build_artifacts/
│   ├── vector_ops.obj              # Compiled CUDA kernel
│   ├── binding.obj                 # Compiled binding
│   └── planar_cuda.pyd             # Final Python module
├── scripts/
│   └── build_final.ps1             # Build automation script
├── tests/
│   └── cuda_tests/
│       └── test_phase1_pipeline.py # Integration tests
└── docs/
    └── __binding__.md              # This file
```

---

## Troubleshooting

### Issue 1: "No module named 'planar_cuda'"

**Cause:** Module not in Python path or not built.

**Solution:**
```python
import sys
sys.path.insert(0, r'D:\...\build_artifacts')  # Add build output path
```

### Issue 2: "找不到指定的模組" (Module not found - Windows)

**Cause:** CUDA DLLs not in PATH.

**Solution:**
```python
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
```

### Issue 3: "57 unresolved external symbols"

**Cause:** Missing C runtime libraries during linking.

**Solution:** Ensure all linker flags are present:
```powershell
-Xlinker "legacy_stdio_definitions.lib" `
-Xlinker "ucrt.lib" `
-Xlinker "vcruntime.lib"
```

### Issue 4: "CUDA kernel launch failed: invalid configuration argument"

**Cause:** Launching kernel with 0 threads (empty vectors).

**Solution:** Add early return in host function:
```cpp
if (n == 0) {
    return std::vector<int>();
}
```

### Issue 5: CMake "No CUDA toolset found"

**Cause:** Visual Studio 2022 not detecting CUDA integration.

**Solution:** Use direct nvcc compilation (build_final.ps1) instead of CMake.

### Issue 6: "Python.h: No such file or directory"

**Cause:** Python development headers not found.

**Solution:**
```powershell
# Verify Python include path
python -c "import sys; print(sys.base_prefix + '\\Include')"
# Add to nvcc command: -I"<output_path>"
```

---

## Advanced Usage

### Adding New CUDA Functions

#### Step 1: Write CUDA Kernel

```cpp
// src/cuda_utils/new_kernel.cu
__global__ void multiply_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

std::vector<float> multiply_gpu(const std::vector<float>& a, const std::vector<float>& b) {
    // Similar to add_vectors_gpu implementation
    // ...
}
```

#### Step 2: Update Binding

```cpp
// src/cpp_binding/binding.cpp
std::vector<float> multiply_gpu(const std::vector<float>& a, const std::vector<float>& b);

PYBIND11_MODULE(planar_cuda, m) {
    // Existing bindings...
    
    // New binding
    m.def("multiply_vectors", &multiply_gpu,
          "Multiply two float vectors using CUDA",
          py::arg("a"), py::arg("b"));
}
```

#### Step 3: Rebuild

```powershell
.\scripts\build_final.ps1
```

#### Step 4: Use in Python

```python
result = planar_cuda.multiply_vectors([1.5, 2.0], [3.0, 4.0])
# Returns: [4.5, 8.0]
```

### Type Conversions

pybind11 automatically converts between C++ and Python types:

| C++ Type | Python Type | Header Required |
|----------|-------------|-----------------|
| `int` | `int` | Default |
| `float` | `float` | Default |
| `std::string` | `str` | Default |
| `std::vector<T>` | `list` | `pybind11/stl.h` |
| `std::map<K,V>` | `dict` | `pybind11/stl.h` |
| `std::tuple<T...>` | `tuple` | `pybind11/stl.h` |

### Performance Considerations

1. **Minimize PCIe Transfers**
   - Upload data once, run multiple kernels
   - Use persistent GPU memory (future: CudaGraph class)

2. **Optimize Kernel Launch**
   ```cpp
   int threads_per_block = 256;  // Multiple of 32 (warp size)
   int blocks = (n + threads_per_block - 1) / threads_per_block;
   ```

3. **Error Checking**
   ```cpp
   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       throw std::runtime_error(cudaGetErrorString(err));
   }
   ```

---

## References

- **pybind11 Documentation**: https://pybind11.readthedocs.io/
- **CUDA C++ Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **CUDA Runtime API**: https://docs.nvidia.com/cuda/cuda-runtime-api/

---

**Document Version:** 1.0  
**Last Updated:** November 28, 2025  
**Status:** Production Ready
