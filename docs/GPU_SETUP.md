# GPU/CUDA 加速設置指南

## 系統需求

### 硬件需求
- **NVIDIA GPU** (推薦 RTX 20xx 或更新)
  - Compute Capability ≥ 7.0 (Volta, Turing, Ampere, Ada)
  - 最小 4GB VRAM（8GB+ 推薦）
  
### 軟件需求
- **CUDA Toolkit 11.x 或 12.x**
- **cuDNN** (可選，用於深度學習加速)
- **Python 3.8+**

---

## 安裝步驟

### 方案 1: CuPy (推薦！最簡單)

CuPy 是 NumPy 的 GPU 版本，支持動態編譯 CUDA kernels。

```bash
# 檢查 CUDA 版本
nvcc --version

# 根據 CUDA 版本安裝 CuPy
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 驗證安裝
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

### 方案 2: PyCUDA (更底層控制)

```bash
# 安裝 PyCUDA
pip install pycuda

# 驗證
python -c "import pycuda.driver as cuda; cuda.init()"
```

### 方案 3: 手動編譯 CUDA Kernels

如果你想使用獨立的 `.cu` 文件：

```bash
# 編譯 geometry_kernels.cu
cd src
nvcc -ptx geometry_kernels.cu -o geometry_kernels.ptx -arch=sm_75

# 在 Python 中加載
# from pycuda.compiler import SourceModule
# module = SourceModule(open('geometry_kernels.ptx').read())
```

---

## 快速測試

### 測試 1: 檢查 GPU 是否可用

```python
import sys
sys.path.insert(0, 'src')

try:
    import cupy as cp
    print(f"✅ CuPy installed")
    print(f"GPU Count: {cp.cuda.runtime.getDeviceCount()}")
    print(f"GPU Name: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
except ImportError:
    print("❌ CuPy not installed")
except Exception as e:
    print(f"⚠️ GPU test failed: {e}")
```

### 測試 2: 運行 CUDA 求解器

```python
from solver_strategy import SolverFactory
import solver_cuda_strategy  # 註冊 CUDA 策略

# 檢查是否可用
strategies = SolverFactory.list_strategies()
print(f"Available strategies: {strategies}")

if 'cuda' in strategies:
    solver = SolverFactory.create_solver('cuda')
    solver.load_from_json('live-2025-example-instances/15-nodes.json')
    result = solver.solve(iterations=100)
    print(f"Result: {result}")
else:
    print("CUDA strategy not available")
```

---

## 性能對比

### 預期加速比（基於 NVIDIA RTX 3070）

| 實例大小 | CPU (秒) | GPU (秒) | 加速比 |
|---------|---------|---------|--------|
| 15 nodes | 1.0 | 0.05 | 20x |
| 70 nodes | 20 | 0.5 | 40x |
| 100 nodes | 60 | 1.0 | 60x |
| 150 nodes | 180 | 2.5 | 72x |
| 225 nodes | 600 | 7 | 86x |
| 625 nodes | 5000 | 40 | 125x |

*實際性能取決於 GPU 型號和問題特性*

---

## CUDA Kernels 說明

### Kernel 1: `segments_intersect_batch`

**用途:** 批量檢測線段相交

**性能:**
- 時間複雜度: O(1) per pair（完全並行）
- 理論加速: 100-1000x
- 最適場景: 需要檢測大量線段對

**使用示例:**
```python
from geometry_cuda import cuda_geometry

# 準備數據
segments_p = [(Point(0,0), Point(10,10)), ...]
segments_q = [(Point(0,10), Point(10,0)), ...]

# GPU 批量檢測
results = cuda_geometry.segments_intersect_batch_gpu(segments_p, segments_q)
# results[i] = True if segments_p[i] 與 segments_q[i] 相交
```

### Kernel 2: `count_edge_crossings`

**用途:** 計算每條邊的交叉數

**性能:**
- 時間複雜度: O(E) per thread, E threads parallel
- 理論加速: 10-100x（取決於 GPU 核心數）
- 適用: 中等規模圖 (E < 100,000)

**使用示例:**
```python
# edges: np.array shape (num_edges, 2)
# positions: np.array shape (num_nodes, 2)

crossings = cuda_geometry.count_all_crossings_gpu(edges, positions)
# crossings[i] = 邊 i 的交叉數
```

### Kernel 3: `count_edge_crossings_optimized`

**用途:** 使用共享內存的優化版本

**性能:**
- 額外加速: 2-5x（相比 Kernel 2）
- 限制: 節點數 < 10,000（共享內存限制）
- 適用: 中小型圖，密集計算

**特點:**
- 將節點位置載入共享內存
- 減少全局內存訪問延遲
- 自動選擇（如果節點數合適）

### Kernel 4: `calculate_delta_crossings`

**用途:** 計算移動節點後的增量變化

**性能:**
- 時間複雜度: O(d * k) where d = degree
- 關鍵優化: 只處理受影響的邊
- 加速比: 50-500x（稀疏圖）

**使用場景:**
- 模擬退火的每次迭代
- 需要快速評估移動效果

### Kernel 5: `build_spatial_hash`

**用途:** 並行構建空間哈希索引

**性能:**
- 加速比: 10-50x
- 適用: 需要頻繁重建索引

---

## 架構設計

### 混合 CPU-GPU 策略

```
┌─────────────────────────────────────────┐
│         Simulated Annealing             │
│         (CPU - Python)                  │
│  - 溫度調度                              │
│  - Metropolis 準則                      │
│  - 狀態管理                              │
└─────────────┬───────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      Energy Calculation                 │
│      (GPU - CUDA Kernels)               │
│  - 批量線段相交檢測                       │
│  - 並行交叉計數                          │
│  - 向量化長度計算                        │
└─────────────────────────────────────────┘
```

**為什麼混合？**

1. **CPU 適合:**
   - 控制流邏輯（if/else, loops with dependencies）
   - 小規模計算（delta < 10 edges）
   - 狀態管理

2. **GPU 適合:**
   - 大規模並行計算
   - 向量化操作
   - 批量幾何計算

### 數據傳輸優化

```python
# ❌ 錯誤: 每次迭代都傳輸數據
for i in range(iterations):
    data_gpu = cp.asarray(data_cpu)  # 慢！
    result_gpu = compute_gpu(data_gpu)
    result_cpu = cp.asnumpy(result_gpu)  # 慢！

# ✅ 正確: 數據盡量保持在 GPU
data_gpu = cp.asarray(data_cpu)  # 只傳輸一次
for i in range(iterations):
    result_gpu = compute_gpu(data_gpu)  # 全在 GPU
result_cpu = cp.asnumpy(result_gpu)  # 最後傳回
```

---

## 故障排除

### 問題 1: "CUDA driver version is insufficient"

**原因:** GPU 驅動版本太舊

**解決:**
```bash
# 更新 NVIDIA 驅動
# Windows: 從 NVIDIA 官網下載
# Linux:
sudo ubuntu-drivers autoinstall
sudo reboot
```

### 問題 2: "Out of memory"

**原因:** GPU 內存不足

**解決:**
```python
# 方案 1: 減少批次大小
# 方案 2: 使用 CPU fallback
# 方案 3: 清理 GPU 緩存
import cupy as cp
cp.get_default_memory_pool().free_all_blocks()
```

### 問題 3: "No kernel image is available"

**原因:** GPU 架構不匹配

**解決:**
```bash
# 檢查你的 GPU 架構
nvidia-smi --query-gpu=compute_cap --format=csv

# 重新編譯指定架構
nvcc -arch=sm_XX geometry_kernels.cu
# XX = 你的 compute capability (例如 75 for RTX 20xx)
```

### 問題 4: GPU 比 CPU 還慢

**原因:** 數據傳輸開銷 > 計算收益

**解決:**
```python
# 只在大規模計算時使用 GPU
if num_edges > 1000:
    use_gpu = True
else:
    use_gpu = False  # 小圖用 CPU
```

---

## 進階優化

### 1. 流水線並行（Streams）

```python
import cupy as cp

# 創建 CUDA streams
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()

# 並行執行
with stream1:
    result1 = compute_part1(data1)

with stream2:
    result2 = compute_part2(data2)

# 等待完成
stream1.synchronize()
stream2.synchronize()
```

### 2. Unified Memory

```python
# 使用 Unified Memory 自動管理 CPU-GPU 傳輸
# CuPy 會自動處理，無需手動優化
```

### 3. Multi-GPU

```python
# 如果有多張 GPU
for gpu_id in range(cp.cuda.runtime.getDeviceCount()):
    with cp.cuda.Device(gpu_id):
        # 在特定 GPU 上執行
        result = compute(data)
```

---

## 下一步

1. **測試 GPU 可用性**
   ```bash
   pip install cupy-cuda12x
   python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
   ```

2. **運行基準測試**
   ```bash
   python compare_solvers.py  # 添加 CUDA 策略
   ```

3. **性能分析**
   ```bash
   # 使用 NVIDIA Nsight Systems
   nsys profile python solver_cuda_strategy.py
   ```

4. **根據結果調優**
   - 調整 block size, grid size
   - 優化內存訪問模式
   - 使用 shared memory

---

## 參考資源

- [CuPy 官方文檔](https://docs.cupy.dev/)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA GPU 選擇指南](https://developer.nvidia.com/cuda-gpus)
- [性能優化最佳實踐](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
