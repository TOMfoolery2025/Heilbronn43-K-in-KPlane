import time
import json
import os
import sys
import networkx as nx
import random
import statistics

# Robust path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from src.LCNv1.api import LCNSolver
except ImportError:
    # Fallback if running from root
    sys.path.append(os.path.join(current_dir, 'src'))
    from LCNv1.api import LCNSolver

def generate_random_graph(num_nodes, num_edges, width=1000, height=1000):
    """Generate a random graph and save to JSON."""
    G = nx.gnm_random_graph(num_nodes, num_edges)
    
    nodes = []
    for i in range(num_nodes):
        nodes.append({
            "id": i,
            "x": random.randint(0, width),
            "y": random.randint(0, height)
        })
        
    edges = []
    for u, v in G.edges():
        edges.append({"source": u, "target": v})
        
    data = {
        "nodes": nodes,
        "edges": edges,
        "width": width,
        "height": height
    }
    
    filename = f"benchmark_graph_{num_nodes}.json"
    with open(filename, 'w') as f:
        json.dump(data, f)
    return filename

def run_benchmark():
    with open('benchmark_results.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"{'Nodes':<10} | {'Strategy':<10} | {'Time (s)':<10} | {'Crossings':<10} | {'Energy':<10}\n")
        f.write("="*60 + "\n")
        
        graph_sizes = [10, 20, 30]
        strategies = ['legacy', 'new']
        iterations = 100
        
        results = {}
        
        for size in graph_sizes:
            num_edges = int(size * 1.5) # Sparse graph
            json_file = generate_random_graph(size, num_edges)
            
            for strategy in strategies:
                times = []
                crossings = []
                energies = []
                
                # Run 3 times to average
                for _ in range(3):
                    try:
                        solver = LCNSolver(strategy=strategy)
                        solver.load_from_json(json_file)
                        
                        start_time = time.time()
                        result = solver.optimize(iterations=iterations)
                        end_time = time.time()
                        
                        times.append(end_time - start_time)
                        crossings.append(result.total_crossings)
                        energies.append(result.energy)
                    except Exception as e:
                        print(f"Error with {strategy} on size {size}: {e}")
                        times.append(0)
                        crossings.append(0)
                        energies.append(0)
                    
                avg_time = statistics.mean(times)
                avg_crossings = statistics.mean(crossings)
                avg_energy = statistics.mean(energies)
                
                line = f"{size:<10} | {strategy:<10} | {avg_time:<10.4f} | {avg_crossings:<10.1f} | {avg_energy:<10.1f}\n"
                print(line.strip())
                f.write(line)
                
                key = (size, strategy)
                results[key] = {
                    'time': avg_time,
                    'crossings': avg_crossings
                }
                
            f.write("-" * 60 + "\n")
            
            # Cleanup
            if os.path.exists(json_file):
                os.remove(json_file)

if __name__ == "__main__":
    run_benchmark()
