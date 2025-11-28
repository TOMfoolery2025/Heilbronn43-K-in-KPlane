#!/usr/bin/env python3
"""
測試更大規模的實例
比較 Legacy vs New 策略在不同規模圖形上的表現
"""
import sys
import time
sys.path.insert(0, 'src')

from solver_strategy import SolverFactory
# 導入策略實現以註冊它們
import solver_legacy_strategy
import solver_new_strategy
import solver_numba_strategy

def test_instance(instance_path, strategy_name, iterations=1000):
    """
    測試單個實例
    
    Returns:
        dict: 測試結果統計
    """
    try:
        solver = SolverFactory.create_solver(strategy_name)
        solver.load_from_json(instance_path)
        
        # 獲取初始狀態
        initial = solver.get_current_stats()
        
        # 計時運行
        start_time = time.time()
        result = solver.solve(iterations=iterations)
        elapsed = time.time() - start_time
        
        # 計算改進
        energy_improvement = initial['energy'] - result['energy']
        improvement_pct = (energy_improvement / initial['energy'] * 100) if initial['energy'] > 0 else 0
        
        return {
            'success': True,
            'initial_energy': initial['energy'],
            'initial_k': initial['k'],
            'initial_crossings': initial['total_crossings'],
            'final_energy': result['energy'],
            'final_k': result['k'],
            'final_crossings': result['total_crossings'],
            'improvement_pct': improvement_pct,
            'elapsed_time': elapsed,
            'iterations': iterations
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def format_number(n):
    """格式化數字顯示"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return str(int(n))

def print_separator():
    """打印分隔線"""
    print("=" * 100)

def main():
    # 測試實例列表
    instances = [
        ('15-nodes.json', 15, 500),
        ('70-nodes.json', 70, 1000),
        ('100-nodes.json', 100, 1000),
        ('150-nodes.json', 150, 1500),
        ('225-nodes.json', 225, 2000),
        # ('625-nodes.json', 625, 3000),  # 可選：非常大的實例
    ]
    
    strategies = ['numba', 'legacy']  # 只測試快的策略
    
    print_separator()
    print(f"{'LARGE INSTANCE TESTING':^100}")
    print_separator()
    print()
    
    # 存儲所有結果
    all_results = {}
    
    # 測試每個實例
    for filename, nodes, iterations in instances:
        instance_path = f'live-2025-example-instances/{filename}'
        print(f"\n📊 Testing: {filename} ({nodes} nodes, {iterations} iterations)")
        print("-" * 100)
        
        all_results[filename] = {}
        
        for strategy in strategies:
            print(f"\n  🔧 Strategy: {strategy.upper()}")
            result = test_instance(instance_path, strategy, iterations)
            
            if result['success']:
                all_results[filename][strategy] = result
                
                print(f"     Initial  → Energy: {format_number(result['initial_energy']):>8}, "
                      f"K: {result['initial_k']:>3}, Crossings: {format_number(result['initial_crossings']):>8}")
                print(f"     Final    → Energy: {format_number(result['final_energy']):>8}, "
                      f"K: {result['final_k']:>3}, Crossings: {format_number(result['final_crossings']):>8}")
                print(f"     Improvement: {result['improvement_pct']:>5.1f}% in {result['elapsed_time']:.2f}s")
            else:
                print(f"     ❌ Error: {result['error']}")
                all_results[filename][strategy] = result
        
        print()
    
    # 打印綜合比較表
    print_separator()
    print(f"{'COMPREHENSIVE COMPARISON':^100}")
    print_separator()
    print()
    
    # 表頭
    header = f"{'Instance':<15} {'Nodes':>6} {'Strategy':>10} | {'Init K':>7} {'Final K':>8} {'Final X':>10} | {'Improve%':>9} {'Time(s)':>9} {'Winner':>8}"
    print(header)
    print("-" * 100)
    
    # 數據行
    for filename, nodes, _ in instances:
        if filename not in all_results:
            continue
            
        results = all_results[filename]
        
        # Legacy 行
        if 'legacy' in results and results['legacy']['success']:
            r = results['legacy']
            print(f"{filename:<15} {nodes:>6} {'Legacy':>10} | "
                  f"{r['initial_k']:>7} {r['final_k']:>8} {format_number(r['final_crossings']):>10} | "
                  f"{r['improvement_pct']:>8.1f}% {r['elapsed_time']:>9.2f}", end='')
            
            # 判斷勝者
            if 'new' in results and results['new']['success']:
                if results['legacy']['final_energy'] < results['new']['final_energy']:
                    print(f" {'✅':>8}")
                else:
                    print(f" {'':>8}")
            else:
                print()
        
        # New 行
        if 'new' in results and results['new']['success']:
            r = results['new']
            print(f"{'':<15} {nodes:>6} {'New':>10} | "
                  f"{r['initial_k']:>7} {r['final_k']:>8} {format_number(r['final_crossings']):>10} | "
                  f"{r['improvement_pct']:>8.1f}% {r['elapsed_time']:>9.2f}", end='')
            
            # 判斷勝者
            if 'legacy' in results and results['legacy']['success']:
                if results['new']['final_energy'] < results['legacy']['final_energy']:
                    print(f" {'✅':>8}")
                else:
                    print(f" {'':>8}")
            else:
                print()
        
        print()
    
    # 統計總結
    print_separator()
    print(f"{'SUMMARY STATISTICS':^100}")
    print_separator()
    print()
    
    legacy_wins = 0
    new_wins = 0
    total_comparisons = 0
    
    for filename in all_results:
        results = all_results[filename]
        if 'legacy' in results and 'new' in results:
            if results['legacy']['success'] and results['new']['success']:
                total_comparisons += 1
                if results['legacy']['final_energy'] < results['new']['final_energy']:
                    legacy_wins += 1
                else:
                    new_wins += 1
    
    if total_comparisons > 0:
        print(f"Total Comparisons: {total_comparisons}")
        print(f"Legacy Wins: {legacy_wins} ({legacy_wins/total_comparisons*100:.1f}%)")
        print(f"New Wins: {new_wins} ({new_wins/total_comparisons*100:.1f}%)")
        print()
        
        if new_wins > legacy_wins:
            print(f"🏆 Overall Winner: NEW ARCHITECTURE")
        elif legacy_wins > new_wins:
            print(f"🏆 Overall Winner: LEGACY")
        else:
            print(f"🤝 Overall Result: TIE")
    
    print_separator()
    
    # 性能分析
    print()
    print(f"{'PERFORMANCE ANALYSIS':^100}")
    print_separator()
    print()
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} Strategy:")
        
        avg_improvement = 0
        avg_time = 0
        avg_k_reduction = 0
        count = 0
        
        for filename in all_results:
            results = all_results[filename]
            if strategy in results and results[strategy]['success']:
                r = results[strategy]
                avg_improvement += r['improvement_pct']
                avg_time += r['elapsed_time']
                k_reduction = r['initial_k'] - r['final_k']
                avg_k_reduction += k_reduction
                count += 1
        
        if count > 0:
            print(f"  Average Improvement: {avg_improvement/count:.1f}%")
            print(f"  Average Time: {avg_time/count:.2f}s")
            print(f"  Average K Reduction: {avg_k_reduction/count:.1f}")
            print(f"  Instances Tested: {count}")
    
    print()
    print_separator()

if __name__ == '__main__':
    main()
