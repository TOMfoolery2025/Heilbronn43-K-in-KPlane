#!/usr/bin/env python3
"""
比較所有求解器策略性能
Legacy vs New vs Numba
"""
import sys
sys.path.insert(0, 'src')

from solver_strategy import SolverFactory
import solver_legacy_strategy
import solver_new_strategy
import solver_numba_strategy

import time

print("=" * 80)
print(" " * 20 + "求解器策略性能測試")
print("=" * 80)

# 測試實例
test_file = 'live-2025-example-instances/15-nodes.json'
iterations = 500

# 可用策略
strategies = SolverFactory.list_strategies()
print(f"\n可用策略: {strategies}")

results = {}

# 測試每個策略
for strategy_name in ['legacy', 'new', 'numba']:
    if strategy_name not in strategies:
        print(f"\n⚠️ {strategy_name} 策略不可用，跳過")
        continue
    
    print(f"\n{'='*80}")
    print(f"測試策略: {strategy_name.upper()}")
    print(f"{'='*80}")
    
    solver = SolverFactory.create_solver(strategy_name)
    solver.load_from_json(test_file)
    
    # 初始狀態
    initial = solver.get_current_stats()
    print(f"初始狀態: E={initial['energy']:,.0f}, K={initial['k']}, X={initial['total_crossings']}")
    
    # 運行
    start = time.time()
    result = solver.solve(iterations=iterations)
    elapsed = time.time() - start
    
    # 結果
    improvement = (initial['energy'] - result['energy']) / initial['energy'] * 100
    
    results[strategy_name] = {
        'initial': initial,
        'final': result,
        'time': elapsed,
        'improvement': improvement
    }
    
    print(f"\n最終狀態: E={result['energy']:,.0f}, K={result['k']}, X={result['total_crossings']}")
    print(f"改進: {improvement:.1f}%")
    print(f"時間: {elapsed:.2f}s")
    print(f"接受率: {result.get('acceptance_rate', 0)*100:.1f}%")

# 綜合比較
print(f"\n{'='*80}")
print(" " * 30 + "綜合比較")
print(f"{'='*80}")

print(f"\n{'策略':<10} {'初始K':>8} {'最終K':>8} {'最終X':>8} {'改進%':>8} {'時間(s)':>10} {'速度(it/s)':>12}")
print("-" * 80)

for name in ['legacy', 'new', 'numba']:
    if name not in results:
        continue
    
    r = results[name]
    speed = iterations / r['time']
    
    print(f"{name:<10} "
          f"{r['initial']['k']:>8} "
          f"{r['final']['k']:>8} "
          f"{r['final']['total_crossings']:>8} "
          f"{r['improvement']:>7.1f}% "
          f"{r['time']:>10.2f} "
          f"{speed:>12.1f}")

# 勝者
print(f"\n{'='*80}")
print("勝者分析:")
print(f"{'='*80}")

best_quality = min(results.items(), key=lambda x: x[1]['final']['energy'])
best_speed = min(results.items(), key=lambda x: x[1]['time'])

print(f"🏆 最佳質量: {best_quality[0].upper()} (能量 = {best_quality[1]['final']['energy']:,.0f})")
print(f"⚡ 最快速度: {best_speed[0].upper()} (時間 = {best_speed[1]['time']:.2f}s)")

# Numba 加速比
if 'numba' in results and 'legacy' in results:
    speedup = results['legacy']['time'] / results['numba']['time']
    print(f"🚀 Numba vs Legacy 加速比: {speedup:.1f}x")

if 'numba' in results and 'new' in results:
    speedup = results['new']['time'] / results['numba']['time']
    print(f"🚀 Numba vs New 加速比: {speedup:.1f}x")

print(f"\n{'='*80}")
