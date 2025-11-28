#!/usr/bin/env python3
"""GPU 環境檢測"""
import cupy as cp

print("✅ CuPy 安裝成功")
print(f"GPU 數量: {cp.cuda.runtime.getDeviceCount()}")

dev_props = cp.cuda.runtime.getDeviceProperties(0)
print(f"GPU 名稱: {dev_props['name'].decode()}")
print(f"計算能力: {dev_props['major']}.{dev_props['minor']}")
print(f"顯存: {dev_props['totalGlobalMem'] / 1024**3:.1f} GB")
print(f"CUDA 版本: {cp.cuda.runtime.runtimeGetVersion()}")

# 簡單性能測試
import time
import numpy as np

n = 10000
print(f"\n簡單性能測試 (矩陣乘法 {n}x{n}):")

# CPU
a_cpu = np.random.rand(n, n)
b_cpu = np.random.rand(n, n)
start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

# GPU
a_gpu = cp.asarray(a_cpu)
b_gpu = cp.asarray(b_cpu)
start = time.time()
c_gpu = a_gpu @ b_gpu
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start
print(f"GPU: {gpu_time:.3f}s")

print(f"加速比: {cpu_time/gpu_time:.1f}x")
