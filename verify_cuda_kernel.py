import os
import sys

# Robustly setup CUDA paths for Windows
def _setup_cuda_paths():
    # Common default path
    default_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    
    # Try to find if not in default
    if not os.path.exists(default_path):
        # Search in Program Files
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base):
            versions = os.listdir(base)
            if versions:
                # Pick the last one (likely newest)
                default_path = os.path.join(base, versions[-1], "bin")

    if os.path.exists(default_path):
        # Add to PATH
        if default_path not in os.environ["PATH"]:
            os.environ["PATH"] += ";" + default_path
        
        # Add to DLL Directory (Python 3.8+)
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(default_path)
                print(f"Added {default_path} to DLL directory")
            except Exception:
                pass
        
        # Set CUDA_PATH if missing
        if "CUDA_PATH" not in os.environ:
            os.environ["CUDA_PATH"] = os.path.dirname(os.path.dirname(default_path))
            print(f"Set CUDA_PATH to {os.environ['CUDA_PATH']}")

_setup_cuda_paths()

import ctypes
try:
    bin_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    
    # Try builtins first
    builtins_path = os.path.join(bin_dir, "nvrtc-builtins64_128.dll")
    if os.path.exists(builtins_path):
        try:
            ctypes.CDLL(builtins_path)
            print(f"SUCCESS: Loaded {builtins_path}")
        except Exception as e:
            print(f"FAILURE: Could not load {builtins_path}: {e}")
            
    # Try nvrtc
    dll_path = os.path.join(bin_dir, "nvrtc64_120_0.dll")
    ctypes.CDLL(dll_path)
    print(f"SUCCESS: Loaded {dll_path}")
except Exception as e:
    print(f"FAILURE: Could not load {dll_path}: {e}")

import cupy as cp

try:
    # Simple Kernel
    code = r'''
    extern "C" __global__
    void test_kernel(float* x) {
        int i = threadIdx.x;
        x[i] = x[i] * 2.0;
    }
    '''
    kernel = cp.RawKernel(code, 'test_kernel')
    
    x = cp.array([1.0, 2.0, 3.0], dtype=cp.float32)
    kernel((1,), (3,), (x,))
    
    print(f"Result: {x}")
    print("SUCCESS: Kernel compiled and ran!")
except Exception as e:
    print(f"FAILURE: {e}")
