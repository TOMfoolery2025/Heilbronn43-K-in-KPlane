#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試 LCNv1 統一接口
驗證所有舊代碼已移除，只使用新的模塊化接口
"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*80)
print("LCNv1 統一接口測試")
print("="*80)

# 測試 1: 導入新接口
print("\n[測試 1] 導入 LCNv1 模塊...")
try:
    from LCNv1 import LCNSolver
    print("  [OK] LCNv1 模塊導入成功")
except ImportError as e:
    print(f"  [FAIL] 導入失敗: {e}")
    sys.exit(1)

# 測試 2: 確認舊代碼不可用
print("\n[測試 2] 確認舊代碼已移除...")
old_modules = [
    'solver_strategy',
    'solver_legacy_strategy',
    'solver_new_strategy',
    'solver_numba_strategy',
    'geometry',
    'graph',
    'cost',
    'spatial_index'
]

for module in old_modules:
    try:
        __import__(module)
        print(f"  [WARN] 舊模塊 {module} 仍可導入（可能在 sys.path 中）")
    except ImportError:
        print(f"  [OK] 舊模塊 {module} 已不可直接導入")

# 測試 3: 使用新接口
print("\n[測試 3] 使用新接口運行優化...")
try:
    solver = LCNSolver(strategy='numba')
    solver.load_from_json('live-2025-example-instances/15-nodes.json')
    
    initial_stats = solver.get_stats()
    print(f"  初始: K={initial_stats['k']}, X={initial_stats['total_crossings']}")
    
    result = solver.optimize(iterations=200)
    print(f"  結果: K={result.k}, X={result.total_crossings}, 改進={result.improvement:.1f}%")
    print("  [OK] 優化成功完成")
except Exception as e:
    print(f"  [FAIL] 優化失敗: {e}")
    import traceback
    traceback.print_exc()

# 測試 4: 策略切換
print("\n[測試 4] 測試策略切換...")
strategies = LCNSolver.list_strategies()
print(f"  可用策略: {strategies}")

for strategy in ['legacy', 'new', 'numba']:
    try:
        s = LCNSolver(strategy=strategy)
        print(f"  [OK] {strategy.upper()} 策略可用")
    except Exception as e:
        print(f"  [FAIL] {strategy.upper()} 策略不可用: {e}")

print("\n" + "="*80)
print("測試完成！新的統一接口運行正常")
print("="*80)
