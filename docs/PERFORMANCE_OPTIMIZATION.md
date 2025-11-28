# 性能優化方案

## 當前瓶頸分析

### 主要性能問題
1. **線段相交檢測** - 每次迭代需要大量幾何計算
2. **空間哈希查詢** - Python 字典操作較慢
3. **Python 循環** - 解釋器開銷大
4. **完全重建空間哈希** - 每次 `calculate()` 都重建

### 時間複雜度
- 初始計算: O(E²) 或 O(E·k) with spatial hash
- Delta 更新: O(d·k) where d = node degree, k = avg crossings
- 迭代次數: 1000-3000 次
- 總計: O(iterations × d × k)

## 優化方案

### 🚀 方案 1: Numba JIT 加速 (推薦！)

**優點:**
- ✅ 無需 GPU，CPU 加速即可
- ✅ 最小代碼改動
- ✅ 10-100x 速度提升
- ✅ 與現有代碼兼容

**實現:**

```python
# 安裝
pip install numba

# 加速幾何運算
from numba import jit

@jit(nopython=True)
def cross_product_fast(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y):
    """JIT 編譯的叉積"""
    dx1 = p1_x - p0_x
    dy1 = p1_y - p0_y
    dx2 = p2_x - p0_x
    dy2 = p2_y - p0_y
    return dx1 * dy2 - dy1 * dx2

@jit(nopython=True)
def segments_intersect_fast(p1_x, p1_y, p2_x, p2_y, 
                            q1_x, q1_y, q2_x, q2_y):
    """JIT 編譯的線段相交檢測"""
    # 使用 cross_product_fast 實現
    # ... (具體實現)
    pass

@jit(nopython=True, parallel=True)
def count_all_crossings(edges, positions, num_edges):
    """並行計算所有交叉"""
    crossings = np.zeros(num_edges, dtype=np.int32)
    # ... 並行循環
    return crossings
```

**預期提升:** 20-50x 在 CPU 上

---

### 🎮 方案 2: CuPy GPU 加速 (中等難度)

**優點:**
- ✅ 100-1000x 速度提升（如有 GPU）
- ✅ NumPy-like API，易於遷移
- ✅ 自動內存管理

**缺點:**
- ❌ 需要 NVIDIA GPU
- ❌ 需要 CUDA 安裝
- ❌ 數據傳輸開銷

**實現:**

```python
# 安裝
pip install cupy-cuda12x  # 根據你的 CUDA 版本

import cupy as cp

class GPUSpatialHash:
    """GPU 加速的空間哈希"""
    
    def __init__(self):
        self.edges_gpu = None
        self.positions_gpu = None
    
    def build(self, edges, positions):
        """將數據傳到 GPU"""
        self.edges_gpu = cp.asarray(edges)
        self.positions_gpu = cp.asarray(positions)
    
    def count_crossings_gpu(self):
        """GPU kernel 計算交叉"""
        # 使用 CuPy 的向量化操作
        # 或自定義 CUDA kernel
        pass
```

**預期提升:** 50-200x（需要 GPU）

---

### ⚡ 方案 3: 混合優化（最佳性能）

結合多種技術：

```python
# 1. Numba JIT 核心計算
# 2. NumPy 向量化
# 3. 增量空間哈希更新
# 4. 多進程並行（對於極大圖）

class OptimizedSolver:
    """
    混合優化策略:
    - Numba JIT 用於幾何計算
    - NumPy arrays 替代 Python lists
    - 增量更新空間哈希（不完全重建）
    - Cython 用於熱點循環
    """
    pass
```

---

## 立即可用的優化（無需新庫）

### 優化 1: 增量空間哈希更新

**當前問題:** 每次 `_ensure_spatial_hash()` 都完全重建

**解決方案:**

```python
class SoftMaxCost:
    def __init__(self, ...):
        self._spatial_hash_dirty = True
        self._last_state_hash = None
    
    def _ensure_spatial_hash(self, graph, state):
        # 只在必要時重建
        state_hash = id(state.positions)  # 簡單版本
        if self._last_state_hash != state_hash or self._spatial_hash_dirty:
            # 重建
            self._spatial_hash = SpatialHash(...)
            # ...
            self._last_state_hash = state_hash
            self._spatial_hash_dirty = False
```

### 優化 2: NumPy 數組代替列表

**當前問題:** `edge_crossings = [0] * graph.num_edges` 使用 Python list

**解決方案:**

```python
import numpy as np

edge_crossings = np.zeros(graph.num_edges, dtype=np.int32)
# NumPy 操作比 Python list 快 10-100x
```

### 優化 3: 緩存常用查詢

```python
class SoftMaxCost:
    def __init__(self, ...):
        self._incident_edges_cache = {}
    
    def _get_incident_edges(self, graph, node_id):
        if node_id not in self._incident_edges_cache:
            self._incident_edges_cache[node_id] = graph.get_incident_edges(node_id)
        return self._incident_edges_cache[node_id]
```

---

## 推薦實施順序

### Phase 1: 快速優化（1 小時）
1. ✅ 將 lists 改為 NumPy arrays
2. ✅ 增量空間哈希更新
3. ✅ 緩存 incident edges

**預期:** 2-5x 提升

### Phase 2: Numba JIT（2-3 小時）
1. ✅ JIT 幾何函數
2. ✅ JIT 交叉計數循環
3. ✅ 並行化獨立計算

**預期:** 10-50x 提升（累計）

### Phase 3: GPU（可選，1 天）
1. ⚠️ 僅在有 NVIDIA GPU 時
2. ⚠️ CuPy 或 PyTorch
3. ⚠️ Custom CUDA kernels

**預期:** 50-500x 提升（如有合適 GPU）

---

## 實際測試數據預估

基於當前性能（15 nodes, 500 iterations = 1s）:

| 實例 | 當前時間 | Phase 1 | Phase 2 | Phase 3 (GPU) |
|------|---------|---------|---------|---------------|
| 15 nodes | 1s | 0.3s | 0.05s | 0.01s |
| 70 nodes | ~20s | 5s | 1s | 0.1s |
| 100 nodes | ~60s | 15s | 3s | 0.3s |
| 150 nodes | ~180s | 40s | 8s | 0.8s |
| 225 nodes | ~600s | 150s | 25s | 2s |
| 625 nodes | ~5000s | 1200s | 200s | 15s |

---

## 立即行動計劃

### 選項 A: 最快實現（推薦！）

```bash
# 安裝 Numba
pip install numba

# 創建加速版本
# 我可以幫你實現 solver_numba_strategy.py
```

這樣可以在 **30 分鐘內** 獲得 10-30x 速度提升！

### 選項 B: 無依賴優化

純 Python/NumPy 優化，無需新庫
- 2-5x 提升
- 完全兼容現有代碼

### 選項 C: GPU 全力版

需要 NVIDIA GPU + CUDA
- 50-500x 提升
- 需要 1-2 天開發時間

---

## 你想要哪個方案？

**我的建議:** 
1. **立即:** 方案 1 (Numba JIT) - 30 分鐘，10-30x 提升
2. **之後:** 如果還不夠快，考慮 GPU

你有 NVIDIA GPU 嗎？你希望我先實現哪個方案？
