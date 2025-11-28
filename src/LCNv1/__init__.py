"""
LCNv1 - Local Crossing Number Minimization System
==================================================

高性能圖形佈局優化系統，專注於最小化局部交叉數 (LCN)。

核心特性:
- 純整數幾何運算
- 多種求解策略 (Legacy, New, Numba JIT)
- 空間哈希加速
- 精確的增量更新
- 完整的測試覆蓋

快速開始:
---------

>>> from LCNv1 import LCNSolver
>>> 
>>> # 創建求解器 (默認使用 Numba JIT 策略)
>>> solver = LCNSolver()
>>> 
>>> # 或指定策略
>>> solver = LCNSolver(strategy='numba')  # 'legacy', 'new', 'numba'
>>> 
>>> # 加載圖形
>>> solver.load_from_json('graph.json')
>>> 
>>> # 優化
>>> result = solver.optimize(iterations=1000)
>>> 
>>> # 導出結果
>>> solver.export_to_json('output.json')
>>> 
>>> # 獲取統計信息
>>> stats = solver.get_stats()
>>> print(f"K = {stats['k']}, Crossings = {stats['total_crossings']}")

策略說明:
---------

- **legacy**: 原始 NumPy 實現，快速但優化效果有限
- **new**: TDD 架構實現，優化效果好但速度較慢  
- **numba**: Numba JIT 編譯，速度快且優化效果好 (推薦)

版本: 1.0.0
作者: TDD Development Team
"""

from .api import LCNSolver, OptimizationResult
from .strategies import StrategyType

__version__ = "1.0.0"
__all__ = ['LCNSolver', 'OptimizationResult', 'StrategyType']
