#!/usr/bin/env python3
"""
測試 CUDA 幾何核心
"""
import sys
sys.path.insert(0, 'src')

import cupy as cp
import numpy as np
import time
from geometry import Point, GeometryCore

print("=" * 60)
print("CUDA 幾何核心測試")
print("=" * 60)

# 測試 1: 編譯 CUDA kernel
print("\n測試 1: 編譯線段相交 CUDA kernel...")

segments_intersect_kernel = cp.RawKernel(r'''
extern "C" __global__
void segments_intersect_batch(
    const int* p1_x, const int* p1_y,
    const int* p2_x, const int* p2_y,
    const int* q1_x, const int* q1_y,
    const int* q2_x, const int* q2_y,
    int* results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int p1x = p1_x[idx], p1y = p1_y[idx];
    int p2x = p2_x[idx], p2y = p2_y[idx];
    int q1x = q1_x[idx], q1y = q1_y[idx];
    int q2x = q2_x[idx], q2y = q2_y[idx];
    
    // 叉積計算
    auto cross = [](int ox, int oy, int ax, int ay, int bx, int by) -> long long {
        long long dx1 = (long long)(ax - ox);
        long long dy1 = (long long)(ay - oy);
        long long dx2 = (long long)(bx - ox);
        long long dy2 = (long long)(by - oy);
        return dx1 * dy2 - dy1 * dx2;
    };
    
    // 檢查端點重疊
    if ((p1x == q1x && p1y == q1y) || (p1x == q2x && p1y == q2y) ||
        (p2x == q1x && p2y == q1y) || (p2x == q2x && p2y == q2y)) {
        results[idx] = 0;
        return;
    }
    
    long long cp1 = cross(p1x, p1y, p2x, p2y, q1x, q1y);
    long long cp2 = cross(p1x, p1y, p2x, p2y, q2x, q2y);
    long long cq1 = cross(q1x, q1y, q2x, q2y, p1x, p1y);
    long long cq2 = cross(q1x, q1y, q2x, q2y, p2x, p2y);
    
    results[idx] = ((cp1 * cp2 < 0) && (cq1 * cq2 < 0)) ? 1 : 0;
}
''', 'segments_intersect_batch')

print("✅ CUDA kernel 編譯成功")

# 測試 2: 批量線段相交檢測
print("\n測試 2: 批量線段相交檢測...")

n_tests = 100000
print(f"生成 {n_tests} 對隨機線段...")

# 生成隨機線段
np.random.seed(42)
p1_x = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
p1_y = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
p2_x = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
p2_y = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
q1_x = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
q1_y = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
q2_x = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)
q2_y = np.random.randint(-1000, 1000, n_tests, dtype=np.int32)

# CPU 版本
print(f"\nCPU: 計算 {n_tests} 次相交檢測...")
start = time.time()
cpu_results = []
for i in range(n_tests):
    p1 = Point(int(p1_x[i]), int(p1_y[i]))
    p2 = Point(int(p2_x[i]), int(p2_y[i]))
    q1 = Point(int(q1_x[i]), int(q1_y[i]))
    q2 = Point(int(q2_x[i]), int(q2_y[i]))
    cpu_results.append(GeometryCore.segments_intersect(p1, p2, q1, q2))
cpu_time = time.time() - start
cpu_results = np.array(cpu_results, dtype=bool)
print(f"CPU 時間: {cpu_time:.3f}s")
print(f"相交數: {cpu_results.sum()}")

# GPU 版本
print(f"\nGPU: 計算 {n_tests} 次相交檢測...")
# 傳輸到 GPU
p1_x_gpu = cp.asarray(p1_x)
p1_y_gpu = cp.asarray(p1_y)
p2_x_gpu = cp.asarray(p2_x)
p2_y_gpu = cp.asarray(p2_y)
q1_x_gpu = cp.asarray(q1_x)
q1_y_gpu = cp.asarray(q1_y)
q2_x_gpu = cp.asarray(q2_x)
q2_y_gpu = cp.asarray(q2_y)
results_gpu = cp.zeros(n_tests, dtype=cp.int32)

# 配置執行
block_size = 256
grid_size = (n_tests + block_size - 1) // block_size

start = time.time()
segments_intersect_kernel(
    (grid_size,), (block_size,),
    (p1_x_gpu, p1_y_gpu, p2_x_gpu, p2_y_gpu,
     q1_x_gpu, q1_y_gpu, q2_x_gpu, q2_y_gpu,
     results_gpu, n_tests)
)
cp.cuda.Stream.null.synchronize()
gpu_time = time.time() - start

gpu_results = cp.asnumpy(results_gpu).astype(bool)
print(f"GPU 時間: {gpu_time:.3f}s")
print(f"相交數: {gpu_results.sum()}")

# 驗證正確性
matches = (cpu_results == gpu_results).sum()
print(f"\n正確性驗證: {matches}/{n_tests} 匹配 ({matches/n_tests*100:.1f}%)")

# 性能比較
print(f"\n{'='*60}")
print(f"性能總結:")
print(f"{'='*60}")
print(f"CPU 時間: {cpu_time:.3f}s")
print(f"GPU 時間: {gpu_time:.3f}s")
print(f"加速比: {cpu_time/gpu_time:.1f}x")
print(f"{'='*60}")

if matches == n_tests:
    print("\n✅ 所有測試通過！GPU 加速可用！")
else:
    print(f"\n⚠️ 警告: {n_tests - matches} 個結果不匹配")
