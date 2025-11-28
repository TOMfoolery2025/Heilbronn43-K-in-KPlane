# Solver 策略模式使用指南

## 概述

系統現在支持兩種求解器策略，您可以根據需求自由選擇：

1. **Legacy Strategy** - 原有的 numpy 實現
2. **New Strategy** - 新的 TDD 架構實現（推薦）

## 快速開始

### 基本使用

```python
from solver_strategy import SolverFactory

# 方法 1: 使用新架構（推薦）
solver = SolverFactory.create_solver('new')

# 方法 2: 使用舊架構
solver = SolverFactory.create_solver('legacy')

# 加載數據
solver.load_from_json('path/to/graph.json')

# 運行優化
result = solver.solve(iterations=1000)

# 獲取結果
print(f"Final Energy: {result['energy']}")
print(f"Final K: {result['k']}")
print(f"Final Crossings: {result['total_crossings']}")

# 導出結果
solver.export_to_json('output.json')
```

### 自定義參數

#### 新架構策略

```python
solver = SolverFactory.create_solver(
    'new',
    w_cross=100.0,    # 交叉懲罰權重
    w_len=1.0,        # 邊長懲罰權重
    power=2,          # 交叉懲罰指數
    cell_size=50      # 空間哈希單元大小
)

result = solver.solve(
    iterations=1000,
    initial_temp=50.0,      # 初始溫度
    cooling_rate=0.995,     # 降溫率
    reheat_threshold=500    # 重新加熱閾值
)
```

#### 舊架構策略

```python
solver = SolverFactory.create_solver('legacy')

result = solver.solve(
    iterations=1000,
    temp=10.0,           # 初始溫度
    cooling_rate=0.995   # 降溫率
)
```

## 性能比較

基於 15-nodes.json 的測試結果（500 次迭代）：

| 指標 | Legacy | New | 勝者 |
|------|--------|-----|------|
| 最終能量 | 240,624 | **204,284** | ✅ New |
| 最終交叉數 | 293 | **98** | ✅ New |
| 最終 K | 24 | **12** | ✅ New |
| 改進率 | 4.0% | **89.6%** | ✅ New |
| 速度 | **0.08s** | 1.08s | ✅ Legacy |

**結論**: 新架構在優化質量上顯著優於舊架構，雖然速度稍慢但改進效果驚人。

## 策略選擇指南

### 使用 Legacy Strategy 當：

- ✅ 需要最快的執行速度
- ✅ 與現有代碼兼容
- ✅ 已經調優過的舊參數
- ✅ 不需要極致優化質量

### 使用 New Strategy 當：

- ✅ 需要最佳優化結果（推薦）
- ✅ 需要精確的數學計算
- ✅ 處理複雜圖形
- ✅ 需要可擴展的架構
- ✅ 需要詳細的統計信息

## 完整示例

```python
#!/usr/bin/env python3
"""
完整的求解器使用示例
"""
import sys
sys.path.insert(0, 'src')

from solver_strategy import SolverFactory

def optimize_graph(input_file, output_file, strategy='new'):
    """
    優化圖形佈局
    
    Args:
        input_file: 輸入 JSON 文件
        output_file: 輸出 JSON 文件
        strategy: 'new' 或 'legacy'
    """
    # 創建求解器
    if strategy == 'new':
        solver = SolverFactory.create_solver(
            'new',
            w_cross=100.0,
            w_len=1.0,
            power=2
        )
    else:
        solver = SolverFactory.create_solver('legacy')
    
    # 加載數據
    print(f"Loading: {input_file}")
    solver.load_from_json(input_file)
    
    # 顯示初始狀態
    initial = solver.get_current_stats()
    print(f"Initial - Energy: {initial['energy']:,.0f}, "
          f"K: {initial['k']}, Crossings: {initial['total_crossings']}")
    
    # 運行優化
    print("Optimizing...")
    result = solver.solve(iterations=1000)
    
    # 顯示結果
    print(f"Final - Energy: {result['energy']:,.0f}, "
          f"K: {result['k']}, Crossings: {result['total_crossings']}")
    
    improvement = initial['energy'] - result['energy']
    print(f"Improvement: {improvement:,.0f} "
          f"({improvement/initial['energy']*100:.1f}%)")
    
    # 保存結果
    solver.export_to_json(output_file)
    print(f"Saved: {output_file}")

if __name__ == '__main__':
    # 使用新架構優化
    optimize_graph(
        'live-2025-example-instances/15-nodes.json',
        'output-new.json',
        strategy='new'
    )
    
    # 使用舊架構優化（比較）
    optimize_graph(
        'live-2025-example-instances/15-nodes.json',
        'output-legacy.json',
        strategy='legacy'
    )
```

## 高級用法

### 查看可用策略

```python
from solver_strategy import SolverFactory

strategies = SolverFactory.list_strategies()
print(f"Available: {strategies}")  # ['legacy', 'new']
```

### 實時監控

```python
solver = SolverFactory.create_solver('new')
solver.load_from_json('graph.json')

# 分批運行並監控
for batch in range(10):
    result = solver.solve(iterations=100)
    stats = solver.get_current_stats()
    print(f"Batch {batch}: Energy={stats['energy']:,.0f}, "
          f"K={stats['k']}, Crossings={stats['total_crossings']}")
```

### 自定義策略（擴展）

```python
from solver_strategy import ISolverStrategy, SolverFactory

class MyCustomStrategy(ISolverStrategy):
    def load_from_json(self, json_path):
        # 自定義實現
        pass
    
    def solve(self, iterations=1000, **kwargs):
        # 自定義優化邏輯
        pass
    
    def get_current_stats(self):
        # 返回統計信息
        pass
    
    def export_to_json(self, output_path):
        # 導出結果
        pass

# 註冊自定義策略
SolverFactory.register_strategy('custom', MyCustomStrategy)

# 使用自定義策略
solver = SolverFactory.create_solver('custom')
```

## 測試

運行比較測試：

```bash
python compare_solvers.py
```

運行所有測試：

```bash
pytest tests/ -v
```

## 架構優勢

### 新架構 (New Strategy)

✅ **數學正確性**
- 純整數幾何運算
- 精確的增量更新（誤差 = 0.0）
- 無浮點誤差

✅ **性能優化**
- 空間哈希 O(E·k) 查詢
- 增量更新 O(d·k)
- 可擴展到大圖

✅ **代碼質量**
- 46 個測試全通過
- TDD 開發流程
- 清晰的關注點分離

### 舊架構 (Legacy Strategy)

✅ **向後兼容**
- 與現有代碼兼容
- 熟悉的參數
- 快速執行

## 故障排除

### 問題：找不到模塊

```bash
# 確保安裝了依賴
pip install -r requirements.txt
```

### 問題：策略未註冊

```python
# 確保導入了策略模塊
import solver_legacy_strategy
import solver_new_strategy
```

### 問題：內存不足

```python
# 使用較大的 cell_size 減少內存使用
solver = SolverFactory.create_solver('new', cell_size=100)
```

## 參考

- 完整架構文檔: `PROJECT_SUMMARY.md`
- 開發計劃: `src/LCNv1/develope_plan.md`
- 測試用例: `tests/`
- 系統演示: `demo_system.py`
