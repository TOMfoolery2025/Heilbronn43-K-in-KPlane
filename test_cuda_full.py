#!/usr/bin/env python3
"""CUDA 完整測試"""
import sys, os

# Fix DLL loading
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')

print("="*60)
print("CUDA 完整測試")
print("="*60)

import cupy as cp
import time

# GPU Info
device = cp.cuda.Device()
props = cp.cuda.runtime.getDeviceProperties(device.id)
print(f"\nGPU: {props['name'].decode()}")
print(f"Compute: {device.compute_capability}")
print(f"Memory: {props['totalGlobalMem']/1024**3:.1f} GB")

# Test 1: Simple arrays
print("\n[Test 1] Arrays...")
a = cp.array([1,2,3])
print(f"  ✓ {a}")

# Test 2: JIT kernel
print("\n[Test 2] JIT Compilation...")
x = cp.arange(1000)
y = x * 2 + 1
cp.cuda.Stream.null.synchronize()
print(f"  ✓ Kernel executed: {y[:5]}")

# Test 3: Matrix multiply
print("\n[Test 3] Matrix Performance...")
n = 3000
a = cp.random.rand(n, n).astype(cp.float32)
b = cp.random.rand(n, n).astype(cp.float32)
cp.cuda.Stream.null.synchronize()

start = time.time()
c = cp.dot(a, b)
cp.cuda.Stream.null.synchronize()
elapsed = time.time() - start

gflops = 2 * n**3 / elapsed / 1e9
print(f"  Matrix {n}x{n}: {elapsed:.3f}s")
print(f"  Performance: {gflops:.1f} GFLOPS")

# Test 4: Custom kernel
print("\n[Test 4] Custom CUDA Kernel...")
kernel = cp.RawKernel(r'''
extern "C" __global__
void my_kernel(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i * i;
}
''', 'my_kernel')

out = cp.zeros(100, dtype=cp.float32)
kernel((1,), (100,), (out, 100))
cp.cuda.Stream.null.synchronize()
print(f"  ✓ Custom kernel: {out[:5]}")

print("\n" + "="*60)
print("✓ CUDA 完全可用！")
print("="*60)
