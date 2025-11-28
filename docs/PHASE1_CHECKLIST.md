# 🚀 Phase 1 Execution Checklist

## Pre-Build Checklist

### Environment
- [ ] CUDA Toolkit 12.6 installed
  - Command: `nvcc --version`
  - Expected: `release 12.6`
  
- [ ] Visual Studio with C++ tools installed
  - Command: `where cl`
  - Expected: Path to `cl.exe` (MSVC compiler)
  
- [ ] CMake installed (version 3.18+)
  - Command: `cmake --version`
  - Expected: `cmake version 3.18` or higher
  
- [ ] Python 3.10+ with virtual environment
  - Command: `python --version`
  - Expected: `Python 3.10` or higher
  
- [ ] Virtual environment activated
  - Command: Check prompt for `(heilbron-43)`
  - Path: `D:/D_backup/2025/tum/25W/hackthon/Hackathon-Nov-25-Heilbronn43/heilbron-43/Scripts/Activate.ps1`

### Dependencies
- [ ] pybind11 installed
  - Command: `pip show pybind11` OR directory `pybind11/` exists
  - Action: `pip install pybind11` or `git clone https://github.com/pybind/pybind11.git`
  
- [ ] pytest installed
  - Command: `pip show pytest`
  - Action: `pip install pytest`
  
- [ ] numpy installed
  - Command: `pip show numpy`
  - Action: `pip install numpy`

---

## Build Checklist

### Automated Build (Recommended)
- [ ] Run build script
  ```powershell
  .\build_and_test.ps1
  ```
  
- [ ] Check for success message
  - Expected output: `✅ PHASE 1 COMPLETE!`
  
- [ ] Verify module created
  - File: `planar_cuda.pyd` in project root
  - Size: ~50-200 KB

### Manual Build (If automated fails)
- [ ] Clone pybind11 (if using subdirectory method)
  ```powershell
  git clone https://github.com/pybind/pybind11.git
  ```

- [ ] Create build directory
  ```powershell
  mkdir build
  cd build
  ```

- [ ] Configure with CMake
  ```powershell
  cmake .. -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
  ```

- [ ] Build
  ```powershell
  cmake --build . --config Release
  ```

- [ ] Copy module
  ```powershell
  copy Release\planar_cuda.pyd ..
  cd ..
  ```

---

## Testing Checklist

### Import Test
- [ ] Test basic import
  ```powershell
  python -c "import planar_cuda; print(planar_cuda.__version__)"
  ```
  - Expected: `0.1.0-phase1`

- [ ] Test CUDA enabled flag
  ```powershell
  python -c "import planar_cuda; print(planar_cuda.cuda_enabled)"
  ```
  - Expected: `True`

### Functionality Tests
- [ ] Run test suite
  ```powershell
  pytest tests/cuda_tests/test_phase1_pipeline.py -v
  ```

- [ ] Verify test results:
  - [ ] `test_module_import` PASSED
  - [ ] `test_simple_addition` PASSED
  - [ ] `test_large_vectors` PASSED
  - [ ] `test_error_handling` PASSED
  - [ ] `test_empty_vectors` PASSED

### Manual Verification
- [ ] Test vector addition manually
  ```powershell
  python -c "import planar_cuda; print(planar_cuda.add_vectors([1,2,3], [4,5,6]))"
  ```
  - Expected: `[5, 7, 9]`

---

## Troubleshooting Checklist

### Build Errors

#### "nvcc not found"
- [ ] Add CUDA to PATH
  ```powershell
  $env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
  ```
- [ ] Restart PowerShell

#### "Visual Studio not found"
- [ ] Check Visual Studio installation
  - [ ] Open Visual Studio Installer
  - [ ] Verify "Desktop development with C++" workload installed
  - [ ] Note the version (2019 or 2022)
  - [ ] Update CMake command with correct generator

#### "pybind11/pybind11.h not found"
- [ ] Option 1: Install globally
  ```powershell
  pip install pybind11[global]
  ```
  
- [ ] Option 2: Clone to project
  ```powershell
  git clone https://github.com/pybind/pybind11.git
  ```
  
- [ ] Option 3: Modify CMakeLists.txt
  ```cmake
  find_package(pybind11 REQUIRED)  # Instead of add_subdirectory
  ```

#### "CUDA_ARCHITECTURES is empty"
- [ ] Edit `CMakeLists.txt`
- [ ] Set correct compute capability:
  ```cmake
  set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4060
  # 86 for RTX 3090
  # 75 for RTX 2080
  ```

### Runtime Errors

#### "ImportError: DLL load failed"
- [ ] Add CUDA DLL path in Python
  ```python
  import os
  os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
  import planar_cuda
  ```

- [ ] Or add to system PATH permanently

#### "CUDA driver version is insufficient"
- [ ] Update NVIDIA drivers
  - [ ] Visit: https://www.nvidia.com/download/index.aspx
  - [ ] Download latest driver for your GPU

#### Tests fail with CUDA errors
- [ ] Check GPU availability
  ```powershell
  nvidia-smi
  ```
  - Should show RTX 4060 with free memory

- [ ] Check CUDA samples work
  ```powershell
  cd "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v12.6"
  # Try running deviceQuery sample
  ```

---

## Success Criteria

### ✅ Phase 1 Complete When:
- [x] All prerequisites installed
- [ ] Build completes without errors
- [ ] `planar_cuda.pyd` created and importable
- [ ] All 5 tests pass
- [ ] Vector addition works correctly
- [ ] No CUDA runtime errors

### 🎯 Ready for Phase 2 When:
- [ ] Phase 1 checklist complete
- [ ] Understand how data flows: Python → C++ → GPU → C++ → Python
- [ ] Can modify and rebuild without issues
- [ ] Tests run reliably

---

## Next Steps After Completion

1. **Read Phase 2 Plan**
   - File: `CUDA_HYBRID_ROADMAP.md` (Phase 2 section)
   - Goal: Port geometry functions to CUDA

2. **Review Test Results**
   - Analyze test output
   - Understand what each test verifies
   - Check performance (10k elements should be fast)

3. **Experiment**
   - Try different vector sizes
   - Modify `vector_ops.cu` to understand kernel code
   - Add print statements to see execution flow

4. **Prepare for Phase 2**
   - Review `src/LCNv1/core/geometry.py`
   - Understand `cross_product` and `segments_intersect`
   - Read about CUDA `__device__` functions

---

## Quick Commands Reference

```powershell
# Full build and test
.\build_and_test.ps1

# Manual build
mkdir build; cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
cd ..; copy build\Release\planar_cuda.pyd .

# Test import
python -c "import planar_cuda; print(planar_cuda.__version__)"

# Run tests
pytest tests/cuda_tests/test_phase1_pipeline.py -v

# Clean rebuild
rm -r build; rm planar_cuda.pyd
.\build_and_test.ps1
```

---

## Documentation Quick Links

- 📖 `QUICKSTART.md` - One-page quick start
- 🔧 `BUILD_INSTRUCTIONS.md` - Detailed build guide
- 🏗️ `ARCHITECTURE_SUMMARY.md` - System architecture
- 🗺️ `CUDA_HYBRID_ROADMAP.md` - Complete roadmap
- 📋 `develope_plan2_EN.md` - Development philosophy

---

## Getting Help

### If Stuck on Build:
1. Check `BUILD_INSTRUCTIONS.md` troubleshooting section
2. Review CMake output for specific errors
3. Verify all prerequisites are correct versions

### If Stuck on Tests:
1. Run tests individually: `pytest tests/cuda_tests/test_phase1_pipeline.py::test_simple_addition -v`
2. Add debug prints to `vector_ops.cu`
3. Check CUDA error messages carefully

### If Performance Issues:
1. Phase 1 should be fast (< 1 second for 10k elements)
2. Check GPU utilization with `nvidia-smi`
3. Verify CUDA kernel is actually running on GPU

---

**Ready to start? Run: `.\build_and_test.ps1`** 🚀
