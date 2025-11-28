#!/usr/bin/env python3
"""
LCNv1 使用示例
展示如何使用統一的 API 接口
"""

import sys
from pathlib import Path

# 添加 src 到路徑
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1 import LCNSolver, OptimizationResult


def example_basic_usage():
    """基本使用示例"""
    print("=" * 80)
    print("示例 1: 基本使用")
    print("=" * 80)
    
    # 創建求解器 (默認使用 Numba 策略)
    solver = LCNSolver()
    
    # 加載圖形
    solver.load_from_json('live-2025-example-instances/15-nodes.json')
    
    # 獲取初始狀態
    initial = solver.get_stats()
    print(f"\n初始狀態:")
    print(f"  能量: {initial['energy']:,.0f}")
    print(f"  K: {initial['k']}")
    print(f"  交叉數: {initial['total_crossings']}")
    
    # 運行優化
    print(f"\n運行優化 (500 iterations)...")
    result = solver.optimize(iterations=500)
    
    # 顯示結果
    print(f"\n最終結果:")
    print(f"  能量: {result.energy:,.0f}")
    print(f"  K: {result.k}")
    print(f"  交叉數: {result.total_crossings}")
    print(f"  改進: {result.improvement:.1f}%")
    print(f"  時間: {result.time:.2f}s")
    print(f"  接受率: {result.acceptance_rate*100:.1f}%")
    
    # 導出結果
    solver.export_to_json('output-example.json')
    print(f"\n結果已導出到: output-example.json")


def example_strategy_comparison():
    """策略比較示例"""
    print("\n" + "=" * 80)
    print("示例 2: 比較不同策略")
    print("=" * 80)
    
    # 列出可用策略
    strategies = LCNSolver.list_strategies()
    print(f"\n可用策略: {strategies}")
    
    # 測試每個策略
    test_file = 'live-2025-example-instances/15-nodes.json'
    iterations = 500
    
    results = {}
    
    for strategy_name in ['legacy', 'new', 'numba']:
        if strategy_name not in strategies:
            print(f"\n⚠️  策略 '{strategy_name}' 不可用，跳過")
            continue
        
        print(f"\n測試策略: {strategy_name.upper()}")
        print("-" * 40)
        
        # 創建求解器
        solver = LCNSolver(strategy=strategy_name)
        solver.load_from_json(test_file)
        
        # 優化
        result = solver.optimize(iterations=iterations)
        
        # 保存結果
        results[strategy_name] = result
        
        # 顯示結果
        print(f"  K: {result.k}")
        print(f"  交叉數: {result.total_crossings}")
        print(f"  改進: {result.improvement:.1f}%")
        print(f"  時間: {result.time:.2f}s")
    
    # 比較結果
    if results:
        print("\n" + "=" * 80)
        print("策略比較總結")
        print("=" * 80)
        
        print(f"\n{'策略':<10} {'K':>5} {'交叉數':>8} {'改進%':>8} {'時間(s)':>10}")
        print("-" * 50)
        
        for name, result in results.items():
            print(f"{name:<10} {result.k:>5} {result.total_crossings:>8} "
                  f"{result.improvement:>7.1f}% {result.time:>10.2f}")
        
        # 找出最佳
        best_quality = min(results.items(), key=lambda x: x[1].energy)
        best_speed = min(results.items(), key=lambda x: x[1].time)
        
        print(f"\n🏆 最佳質量: {best_quality[0].upper()}")
        print(f"⚡ 最快速度: {best_speed[0].upper()}")


def example_custom_parameters():
    """自定義參數示例"""
    print("\n" + "=" * 80)
    print("示例 3: 自定義參數")
    print("=" * 80)
    
    # 創建自定義參數的求解器
    solver = LCNSolver(
        strategy='numba',
        w_cross=100.0,  # 交叉懲罰權重
        w_len=1.0,      # 邊長懲罰權重
        power=2         # 交叉懲罰指數
    )
    
    solver.load_from_json('live-2025-example-instances/15-nodes.json')
    
    # 自定義優化參數
    result = solver.optimize(
        iterations=1000,
        initial_temp=100.0,      # 更高的初始溫度
        cooling_rate=0.99,       # 更慢的降溫
        reheat_threshold=300     # 更早重新加熱
    )
    
    print(f"\n優化結果:")
    print(f"  K: {result.k}")
    print(f"  交叉數: {result.total_crossings}")
    print(f"  改進: {result.improvement:.1f}%")


def example_programmatic_usage():
    """程式化使用示例"""
    print("\n" + "=" * 80)
    print("示例 4: 程式化使用")
    print("=" * 80)
    
    # 批量處理多個文件
    instances = [
        '15-nodes.json',
        '70-nodes.json',
    ]
    
    print(f"\n批量處理 {len(instances)} 個實例...")
    
    for filename in instances:
        filepath = f'live-2025-example-instances/{filename}'
        
        print(f"\n處理: {filename}")
        
        try:
            solver = LCNSolver(strategy='numba')
            solver.load_from_json(filepath)
            
            result = solver.optimize(iterations=500)
            
            print(f"  ✓ K: {result.k}, 交叉數: {result.total_crossings}, "
                  f"改進: {result.improvement:.1f}%, 時間: {result.time:.2f}s")
            
            # 導出結果
            output_name = f"output-{filename}"
            solver.export_to_json(output_name)
            
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")


def main():
    """運行所有示例"""
    print("\n")
    print("=" * 80)
    print(" " * 20 + "LCNv1 API 使用示例")
    print("=" * 80)
    
    # 運行示例
    example_basic_usage()
    example_strategy_comparison()
    example_custom_parameters()
    example_programmatic_usage()
    
    print("\n" + "=" * 80)
    print("所有示例完成！")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
