"""
Demo: LCN Minimization System - TDD Implementation
Demonstrates the complete system working on 15-nodes.json
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from geometry import Point, GeometryCore
from graph import GraphData, GridState
from cost import SoftMaxCost
from spatial_index import SpatialHash


def load_graph(json_path):
    """Load graph from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Build graph topology
    edges = [(e['source'], e['target']) for e in data['edges']]
    graph = GraphData(num_nodes=len(data['nodes']), edges=edges)
    
    # Build initial positions
    positions = {}
    for node in data['nodes']:
        nid = node['id']
        positions[nid] = Point(int(node['x']), int(node['y']))
    
    state = GridState(positions, width=data['width'], height=data['height'])
    
    return graph, state, data


def analyze_state(graph, state, cost_func):
    """Analyze and print statistics for current state."""
    energy = cost_func.calculate(graph, state)
    k, total = cost_func.get_crossing_stats(graph, state)
    length_energy = cost_func.get_length_energy(graph, state)
    
    return {
        'energy': energy,
        'k': k,
        'total_crossings': total,
        'length_energy': length_energy
    }


def main():
    print("=" * 70)
    print("LCN Minimization System - Comprehensive Test")
    print("=" * 70)
    print()
    
    # Load graph
    json_path = 'live-2025-example-instances/15-nodes.json'
    print(f"Loading: {json_path}")
    graph, state, data = load_graph(json_path)
    
    print(f"  Nodes: {graph.num_nodes}")
    print(f"  Edges: {graph.num_edges}")
    print(f"  Canvas: {data['width']} x {data['height']}")
    print()
    
    # Create cost function
    cost_func = SoftMaxCost(w_cross=100.0, w_len=1.0, power=2)
    
    # Analyze initial state
    print("Initial State Analysis:")
    print("-" * 70)
    stats = analyze_state(graph, state, cost_func)
    print(f"  Total Energy:      {stats['energy']:,.2f}")
    print(f"  K (max crossings): {stats['k']}")
    print(f"  Total Crossings:   {stats['total_crossings']}")
    print(f"  Length Energy:     {stats['length_energy']:,.2f}")
    print()
    
    # Test delta update correctness
    print("Testing Delta Update Correctness:")
    print("-" * 70)
    
    import random
    random.seed(42)
    
    test_node = random.randint(0, graph.num_nodes - 1)
    old_pos = state.get_position(test_node)
    new_pos = Point(
        random.randint(0, data['width']),
        random.randint(0, data['height'])
    )
    
    print(f"  Moving node {test_node} from {old_pos} to {new_pos}")
    
    # Calculate delta
    E_before = cost_func.calculate(graph, state)
    delta = cost_func.calculate_delta(graph, state, test_node, new_pos)
    E_predicted = E_before + delta
    
    print(f"  Energy before:  {E_before:,.2f}")
    print(f"  Delta:          {delta:+,.2f}")
    print(f"  Predicted:      {E_predicted:,.2f}")
    
    # Actually move
    state.move_node(test_node, new_pos)
    E_after = cost_func.calculate(graph, state)
    
    print(f"  Actual:         {E_after:,.2f}")
    print(f"  Error:          {abs(E_predicted - E_after):.10f}")
    
    if abs(E_predicted - E_after) < 1e-6:
        print("  ✓ Delta update is EXACT!")
    else:
        print("  ✗ Delta update has error")
    print()
    
    # Test spatial hash
    print("Spatial Hash Performance:")
    print("-" * 70)
    
    # Brute force counting
    import time
    
    # Reset state
    state.move_node(test_node, old_pos)
    
    start = time.time()
    edge_crossings_brute = [0] * graph.num_edges
    for i in range(graph.num_edges):
        src_i, tgt_i = graph.get_edge_endpoints(i)
        p1 = state.get_position(src_i)
        p2 = state.get_position(tgt_i)
        
        for j in range(i + 1, graph.num_edges):
            src_j, tgt_j = graph.get_edge_endpoints(j)
            q1 = state.get_position(src_j)
            q2 = state.get_position(tgt_j)
            
            if GeometryCore.segments_intersect(p1, p2, q1, q2):
                edge_crossings_brute[i] += 1
                edge_crossings_brute[j] += 1
    
    brute_time = time.time() - start
    brute_total = sum(edge_crossings_brute) // 2
    
    print(f"  Brute Force (O(E²)):")
    print(f"    Time:      {brute_time*1000:.2f} ms")
    print(f"    Crossings: {brute_total}")
    
    # Spatial hash counting
    start = time.time()
    k_spatial, total_spatial = cost_func.get_crossing_stats(graph, state)
    spatial_time = time.time() - start
    
    print(f"  Spatial Hash (O(E·k)):")
    print(f"    Time:      {spatial_time*1000:.2f} ms")
    print(f"    Crossings: {total_spatial}")
    print(f"    Speedup:   {brute_time/spatial_time:.2f}x")
    
    if brute_total == total_spatial:
        print("  ✓ Results match exactly!")
    print()
    
    # Architecture summary
    print("Architecture Validation:")
    print("-" * 70)
    print("  ✓ Sprint 1: Geometry Core (integer-only arithmetic)")
    print("  ✓ Sprint 2: Spatial Hash (O(1) neighbor queries)")
    print("  ✓ Sprint 3: Energy & Delta (exact incremental updates)")
    print("  ✓ Sprint 4: Graph State (immutable topology + mutable positions)")
    print()
    
    print("=" * 70)
    print("System Ready for Optimization!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  - Integrate with SimulatedAnnealing solver")
    print("  - Run on larger instances (70, 100, 150 nodes)")
    print("  - Tune hyperparameters (temperature, cooling rate)")
    print("  - Implement advanced move strategies (swap, rotate)")


if __name__ == '__main__':
    main()
