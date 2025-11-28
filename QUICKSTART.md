# Phase 1 Quick Start Guide

## 🚀 ONE-COMMAND BUILD

```powershell
.\build_and_test.ps1
```

This script will:
1. ✅ Check all prerequisites (CUDA, CMake, Python)
2. ✅ Install Python dependencies (pybind11, pytest)
3. ✅ Download pybind11 if missing
4. ✅ Configure with CMake
5. ✅ Build the CUDA module
6. ✅ Run all tests
7. ✅ Report success or failure

---

## 📁 What Was Created

```
Hackathon-Nov-25-Heilbronn43/
├── src/
│   ├── cuda_utils/
│   │   └── vector_ops.cu         ⭐ CUDA kernel (GPU code)
│   └── cpp_binding/
│       └── binding.cpp            ⭐ Python ↔ C++ bridge
├── tests/
│   └── cuda_tests/
│       └── test_phase1_pipeline.py ⭐ 5 comprehensive tests
├── CMakeLists.txt                 ⭐ Build configuration
├── setup.py                       ⭐ Alternative build method
├── build_and_test.ps1             ⭐ Automated build script
├── BUILD_INSTRUCTIONS.md          📖 Detailed instructions
└── CUDA_HYBRID_ROADMAP.md         📖 Full roadmap
```

---

## 🧪 Manual Testing

If you prefer to build manually:

### Step 1: Install pybind11
```powershell
pip install pybind11
# Or clone: git clone https://github.com/pybind/pybind11.git
```

### Step 2: Build
```powershell
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ..
copy build\Release\planar_cuda.pyd .
```

### Step 3: Test
```powershell
# Quick test
python -c "import planar_cuda; print(planar_cuda.add_vectors([1,2], [3,4]))"
# Output: [4, 6]

# Full test suite
pytest tests/cuda_tests/test_phase1_pipeline.py -v
```

---

## ✅ Success Checklist

- [ ] `.\build_and_test.ps1` runs without errors
- [ ] Output shows "✅ PHASE 1 COMPLETE!"
- [ ] All 5 tests pass:
  - [ ] test_module_import
  - [ ] test_simple_addition
  - [ ] test_large_vectors
  - [ ] test_error_handling
  - [ ] test_empty_vectors
- [ ] `import planar_cuda` works in Python
- [ ] Vector addition returns correct results

---

## 🔧 Troubleshooting

### Error: "nvcc not found"
```powershell
# Add CUDA to PATH
$env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
```

### Error: "Visual Studio not found"
Install "Desktop development with C++" workload from Visual Studio Installer

### Error: "pybind11 not found"
```powershell
# Option 1: pip install
pip install pybind11

# Option 2: Git clone
git clone https://github.com/pybind/pybind11.git

# Option 3: Manual download
# https://github.com/pybind/pybind11/archive/refs/heads/master.zip
# Extract to project root
```

### Error: "Cannot import planar_cuda"
```python
# Add CUDA DLL path before import
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
import planar_cuda
```

---

## 📊 What Each File Does

### `vector_ops.cu` (CUDA Kernel)
- Implements GPU vector addition
- Demonstrates CUDA memory management
- Shows kernel launch pattern
- **Learning point**: How to write `__global__` kernels

### `binding.cpp` (pybind11 Interface)
- Exposes C++/CUDA functions to Python
- Handles type conversion (Python list ↔ C++ vector)
- Defines module metadata
- **Learning point**: How to create Python modules from C++

### `test_phase1_pipeline.py` (TDD Tests)
- Verifies module import works
- Tests basic functionality
- Checks edge cases (empty vectors, errors)
- Tests performance (10k elements)
- **Learning point**: Test-driven development pattern

### `CMakeLists.txt` (Build System)
- Configures CUDA compiler
- Links Python libraries
- Sets GPU architecture (compute capability 8.9)
- Manages dependencies
- **Learning point**: How to build hybrid C++/CUDA/Python projects

---

## 🎯 Next Steps After Phase 1

Once all tests pass, you're ready for **Phase 2: Geometry Kernels**:

1. Port `cross_product` to CUDA `__device__` function
2. Port `segments_intersect` to CUDA
3. Create `count_crossings_kernel` 
4. Test against Python implementation using `sol-15-nodes-5-planar.json`

**Goal**: GPU geometry must match Python geometry 100%

---

## 💡 Key Concepts Learned

1. **Hybrid Architecture**: Python (frontend) + C++ (binding) + CUDA (compute)
2. **Memory Transfer**: Host (CPU) ↔ Device (GPU) via `cudaMemcpy`
3. **Kernel Launch**: `kernel<<<blocks, threads>>>(args)`
4. **Error Handling**: Check `cudaGetLastError()` and `cudaDeviceSynchronize()`
5. **Python Binding**: Use `pybind11` to expose C++ to Python

---

## 📈 Performance Expectations

Phase 1 baseline (vector addition):
- **CPU (Python)**: ~1 ms for 10k elements
- **GPU (CUDA)**: ~0.5 ms for 10k elements
- **Speedup**: ~2x (small data, memory transfer overhead dominates)

For larger problems (Phase 4):
- **Expected speedup**: 10-100x for graph optimization
- **Reason**: More computation per byte transferred

---

## 🆘 Getting Help

If stuck:
1. Check `BUILD_INSTRUCTIONS.md` for detailed troubleshooting
2. Review `CUDA_HYBRID_ROADMAP.md` for architecture overview
3. Examine build output in `build/` directory
4. Run tests individually: `pytest tests/cuda_tests/test_phase1_pipeline.py::test_simple_addition -v`

---

## 🎉 Ready to Build?

```powershell
# Run this:
.\build_and_test.ps1

# If successful, you'll see:
# ✅ PHASE 1 COMPLETE!
```

**Then move to Phase 2: Geometry Kernels! 🚀**
