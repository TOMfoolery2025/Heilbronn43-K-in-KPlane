"""
Solver Strategy Comparison Demo
Demonstrates how to use both legacy and new solver strategies.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import strategy implementations (this registers them with factory)
import solver_legacy_strategy
import solver_new_strategy
from solver_strategy import SolverFactory


def run_solver_comparison(json_path: str, iterations: int = 1000):
    """
    Compare legacy and new solver strategies.
    
    Args:
        json_path: Path to input JSON file
        iterations: Number of iterations to run
    """
    print("=" * 70)
    print("Solver Strategy Comparison")
    print("=" * 70)
    print()
    
    strategies_to_test = ['legacy', 'new']
    results = {}
    
    for strategy_name in strategies_to_test:
        print(f"Testing Strategy: {strategy_name.upper()}")
        print("-" * 70)
        
        # Create solver
        solver = SolverFactory.create_solver(strategy_name)
        
        # Load data
        print(f"  Loading: {json_path}")
        solver.load_from_json(json_path)
        
        # Get initial stats
        initial_stats = solver.get_current_stats()
        print(f"  Initial Energy:    {initial_stats['energy']:,.2f}")
        print(f"  Initial K:         {initial_stats['k']}")
        print(f"  Initial Crossings: {initial_stats['total_crossings']}")
        
        # Run optimization
        print(f"  Running {iterations} iterations...")
        start_time = time.time()
        
        result = solver.solve(iterations=iterations)
        
        elapsed_time = time.time() - start_time
        
        # Display results
        print(f"  Final Energy:      {result['energy']:,.2f}")
        print(f"  Final K:           {result['k']}")
        print(f"  Final Crossings:   {result['total_crossings']}")
        
        improvement = initial_stats['energy'] - result['energy']
        improvement_pct = (improvement / initial_stats['energy']) * 100
        
        print(f"  Improvement:       {improvement:,.2f} ({improvement_pct:.1f}%)")
        print(f"  Time:              {elapsed_time:.2f}s")
        
        if 'accepted' in result['stats']:
            acc_rate = result['stats']['acceptance_rate'] * 100
            print(f"  Acceptance Rate:   {acc_rate:.1f}%")
        
        print()
        
        # Store results
        results[strategy_name] = {
            'initial_energy': initial_stats['energy'],
            'final_energy': result['energy'],
            'initial_crossings': initial_stats['total_crossings'],
            'final_crossings': result['total_crossings'],
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'time': elapsed_time,
            'k': result['k']
        }
    
    # Comparison
    print("=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<25} {'Legacy':<20} {'New':<20}")
    print("-" * 70)
    
    metrics = [
        ('Final Energy', 'final_energy', '{:,.0f}'),
        ('Final Crossings', 'final_crossings', '{:d}'),
        ('Final K', 'k', '{:d}'),
        ('Improvement %', 'improvement_pct', '{:.1f}%'),
        ('Time (seconds)', 'time', '{:.2f}'),
    ]
    
    for label, key, fmt in metrics:
        legacy_val = results['legacy'][key]
        new_val = results['new'][key]
        
        print(f"{label:<25} {fmt.format(legacy_val):<20} {fmt.format(new_val):<20}")
    
    print()
    print("=" * 70)
    
    # Winner determination
    if results['new']['final_energy'] < results['legacy']['final_energy']:
        print("Winner: NEW Architecture (better final energy)")
    elif results['legacy']['final_energy'] < results['new']['final_energy']:
        print("Winner: LEGACY (better final energy)")
    else:
        print("Tie: Both achieved same energy")
    
    print("=" * 70)


def demonstrate_strategy_selection():
    """Demonstrate how to select and use different strategies."""
    print("\n" + "=" * 70)
    print("Strategy Selection Demo")
    print("=" * 70)
    print()
    
    # List available strategies
    strategies = SolverFactory.list_strategies()
    print(f"Available strategies: {', '.join(strategies)}")
    print()
    
    # Example 1: Using legacy strategy
    print("Example 1: Using Legacy Strategy")
    print("-" * 70)
    legacy_solver = SolverFactory.create_solver('legacy')
    print(f"Created: {type(legacy_solver).__name__}")
    print()
    
    # Example 2: Using new strategy with custom parameters
    print("Example 2: Using New Strategy with Custom Parameters")
    print("-" * 70)
    new_solver = SolverFactory.create_solver(
        'new',
        w_cross=200.0,  # Higher crossing weight
        w_len=0.5,      # Lower length weight
        power=3,        # Cubic penalty
        cell_size=100   # Larger cells
    )
    print(f"Created: {type(new_solver).__name__}")
    print(f"  Cross Weight: 200.0")
    print(f"  Length Weight: 0.5")
    print(f"  Power: 3")
    print(f"  Cell Size: 100")
    print()


if __name__ == '__main__':
    # Demonstrate strategy selection
    demonstrate_strategy_selection()
    
    # Run comparison on 15-nodes.json
    json_path = 'live-2025-example-instances/15-nodes.json'
    
    if os.path.exists(json_path):
        print("\nRunning solver comparison on 15-nodes.json...")
        print()
        run_solver_comparison(json_path, iterations=500)
    else:
        print(f"\nWarning: {json_path} not found")
        print("Skipping comparison test")
