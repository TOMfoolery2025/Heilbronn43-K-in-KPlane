"""
修復 CuPy NVRTC DLL 加載問題
在導入 CuPy 前設置 DLL 搜索路徑
"""
import os
import sys

# 添加 CUDA 12.6 bin 到 DLL 搜索路徑（Windows 10+）
cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(cuda_bin)
    print(f"[OK] Added DLL directory: {cuda_bin}")

# 也添加到 PATH（向後兼容）
if cuda_bin not in os.environ['PATH']:
    os.environ['PATH'] = cuda_bin + os.pathsep + os.environ['PATH']
    print(f"[OK] Added to PATH: {cuda_bin}")

# 現在導入 CuPy
try:
    import cupy as cp
    print(f"[OK] CuPy {cp.__version__} loaded")
    print(f"[OK] GPU: {cp.cuda.Device().compute_capability}")
    
    # 測試簡單運算
    a = cp.array([1, 2, 3])
    print(f"[OK] Simple array: {a}")
    
    # 測試 JIT 編譯（這會用到 nvrtc）
    print("\n[Testing JIT compilation...]")
    x = cp.arange(10)
    y = cp.arange(10)
    z = x + y * 2  # 簡單 kernel
    print(f"[OK] JIT kernel executed: {z[:5]}")
    
    print("\n[SUCCESS] CuPy with NVRTC is working!")
    
except Exception as e:
    print(f"[FAIL] {e}")
    import traceback
    traceback.print_exc()
