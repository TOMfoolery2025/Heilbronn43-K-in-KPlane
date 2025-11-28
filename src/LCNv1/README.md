# LCNv1 - Local Crossing Number Minimization System

高性能圖形佈局優化系統，專注於最小化局部交叉數 (LCN)。

## 🎯 特性

- ✅ **純整數幾何運算** - 無浮點誤差
- ✅ **多種求解策略** - Legacy, New, Numba JIT
- ✅ **空間哈希加速** - O(E·k) 查詢複雜度
- ✅ **精確增量更新** - 零誤差的 Delta 計算
- ✅ **完整測試覆蓋** - 46 個單元測試
- ✅ **統一 API 接口** - 簡單易用

## 📦 目錄結構

```
src/LCNv1/
├── __init__.py          # 模塊入口，導出公共 API
├── api.py               # LCNSolver 統一接口
├── core/                # 核心模塊
│   ├── geometry.py      # 幾何計算 (Point, GeometryCore)
│   ├── graph.py         # 圖結構 (GraphData, GridState)
│   ├── spatial_index.py # 空間哈希索引
│   └── cost.py          # 代價函數
├── strategies/          # 求解策略
│   ├── base.py          # 抽象接口
│   ├── legacy.py        # 原始 NumPy 實現
│   ├── new.py           # TDD 架構實現
│   ├── numba_jit.py     # Numba JIT 加速
│   └── register.py      # 自動註冊策略
└── tests/               # 單元測試
    ├── test_geometry.py
    ├── test_spatial.py
    ├── test_energy.py
    └── test_solver.py
```

## 🚀 快速開始

### 安裝依賴

```bash
pip install numpy numba pytest
```

### 基本使用

```python
from LCNv1 import LCNSolver

# 創建求解器 (默認使用 Numba 策略)
solver = LCNSolver()

# 加載圖形
solver.load_from_json('input.json')

# 運行優化
result = solver.optimize(iterations=1000)

# 查看結果
print(f"K = {result.k}")
print(f"交叉數 = {result.total_crossings}")
print(f"改進 = {result.improvement:.1f}%")

# 導出結果
solver.export_to_json('output.json')
```

### 選擇策略

```python
# 使用 Legacy 策略 (快速但優化效果一般)
solver = LCNSolver(strategy='legacy')

# 使用 New 策略 (優化效果好但較慢)
solver = LCNSolver(strategy='new')

# 使用 Numba 策略 (推薦：快速且優化效果好)
solver = LCNSolver(strategy='numba')
```

### 自定義參數

```python
solver = LCNSolver(
    strategy='numba',
    w_cross=100.0,  # 交叉懲罰權重
    w_len=1.0,      # 邊長懲罰權重
    power=2         # 交叉懲罰指數
)

result = solver.optimize(
    iterations=1000,
    initial_temp=50.0,
    cooling_rate=0.995,
    reheat_threshold=500
)
```

## 📊 性能對比

基於 15-nodes.json 測試 (500 iterations):

| 策略 | 速度 (it/s) | 最終 K | 交叉數 | 改進% | 推薦度 |
|------|-------------|--------|--------|-------|--------|
| **Legacy** | 7,487 | 24 | 270 | 4% | ⭐ |
| **New** | 488 | 11 | 82 | 87% | ⭐⭐⭐ |
| **Numba** | 9,524 | 8 | 63 | 88% | ⭐⭐⭐⭐⭐ |

**結論**: Numba 策略提供最佳性能和質量組合。

## 📖 API 文檔

### LCNSolver

主要接口類，提供圖形佈局優化功能。

#### 初始化

```python
LCNSolver(
    strategy='numba',   # 求解策略
    w_cross=100.0,      # 交叉懲罰權重
    w_len=1.0,          # 邊長懲罰權重
    power=2             # 交叉懲罰指數
)
```

#### 方法

- **`load_from_json(json_path)`**: 從 JSON 加載圖形
- **`optimize(iterations, ...)`**: 執行優化
- **`get_stats()`**: 獲取當前統計信息
- **`export_to_json(output_path)`**: 導出結果
- **`list_strategies()`** (靜態): 列出可用策略

### OptimizationResult

優化結果數據類。

#### 屬性

- `energy`: 最終能量值
- `k`: 最大交叉數
- `total_crossings`: 總交叉數
- `iterations`: 迭代次數
- `acceptance_rate`: 接受率
- `time`: 運行時間 (秒)
- `improvement`: 改進百分比

## 🧪 運行測試

```bash
# 運行所有測試
pytest src/LCNv1/tests/ -v

# 運行特定測試
pytest src/LCNv1/tests/test_geometry.py -v

# 運行性能測試
python example_usage.py
```

## 📝 輸入格式

JSON 格式:

```json
{
  "nodes": [
    {"id": 0, "x": 100, "y": 200},
    {"id": 1, "x": 300, "y": 150}
  ],
  "edges": [
    {"source": 0, "target": 1}
  ]
}
```

## 🔧 開發

### 添加新策略

1. 創建新文件: `src/LCNv1/strategies/my_strategy.py`
2. 實現 `ISolverStrategy` 接口
3. 在 `register.py` 中註冊

```python
from .base import ISolverStrategy, SolverFactory

class MyStrategy(ISolverStrategy):
    def load_from_json(self, json_path):
        # 實現
        pass
    
    def solve(self, iterations=1000, **kwargs):
        # 實現
        pass
    
    def get_current_stats(self):
        # 實現
        pass
    
    def export_to_json(self, output_path):
        # 實現
        pass

# 註冊
SolverFactory.register_strategy('my_strategy', MyStrategy)
```

## 📚 相關文檔

- `PERFORMANCE_OPTIMIZATION.md` - 性能優化指南
- `GPU_SETUP.md` - GPU 加速設置
- `SOLVER_STRATEGY_GUIDE.md` - 策略使用指南
- `PROJECT_SUMMARY.md` - 項目完整文檔

## 📄 授權

MIT License

## 👥 作者

TDD Development Team - 2025
