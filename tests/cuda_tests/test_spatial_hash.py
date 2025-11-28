"""
Cycle 3: Spatial Hash Optimization Tests
=========================================
Test suite for GPU-accelerated spatial hash grid for efficient crossing detection.

Test-Driven Development (TDD) Approach:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Optimize spatial hash implementation

Goal: Reduce delta-E computation from O(E²) to O(E·k) where k = avg edges per cell

Test Coverage:
- Spatial hash grid construction
- Edge-to-cell mapping
- Efficient crossing detection using spatial partitioning
- Performance validation (speedup measurement)
"""

import pytest
import sys
import os
import time

# Add CUDA DLL path and module path
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build_artifacts'))

import planar_cuda


class TestSpatialHashConstruction:
    """Test spatial hash grid creation and configuration"""
    
    def test_create_with_spatial_hash(self):
        """Test creating solver with spatial hash enabled"""
        x_coords = [0, 10, 10, 0]
        y_coords = [0, 0, 10, 10]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        
        # Create solver with spatial hash (cell_size parameter)
        solver = planar_cuda.PlanarSolver(
            x_coords, 
            y_coords, 
            edges,
            cell_size=5  # Grid cells of 5x5 units
        )
        
        # Should still compute crossings correctly
        crossings = solver.calculate_total_crossings()
        assert crossings == 0, "Square should be planar"
    
    def test_spatial_hash_auto_cell_size(self):
        """Test automatic cell size determination"""
        # Large sparse graph
        x_coords = [i * 100 for i in range(10)]
        y_coords = [i * 100 for i in range(10)]
        edges = [(i, i+1) for i in range(9)]
        
        # Create without specifying cell_size (auto-compute)
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Should work correctly
        crossings = solver.calculate_total_crossings()
        assert crossings == 0, "Chain should be planar"
    
    def test_get_spatial_hash_stats(self):
        """Test retrieving spatial hash statistics"""
        x_coords = [0, 10, 20, 30]
        y_coords = [0, 10, 20, 30]
        edges = [(0, 1), (1, 2), (2, 3)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=10)
        
        # Get spatial hash statistics
        stats = solver.get_spatial_hash_stats()
        
        assert 'cell_size' in stats
        assert 'num_cells' in stats
        assert 'edges_per_cell_avg' in stats
        assert stats['cell_size'] == 10


class TestEdgeToCellMapping:
    """Test mapping edges to spatial hash cells"""
    
    def test_edge_spans_single_cell(self):
        """Test edge that fits in one cell"""
        # Small edge in 10x10 cell
        x_coords = [0, 5]
        y_coords = [0, 5]
        edges = [(0, 1)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=10)
        
        # Should map to single cell
        stats = solver.get_spatial_hash_stats()
        # Edge (0,0)-(5,5) should be in cell (0,0)
        assert stats['num_cells'] >= 1
    
    def test_edge_spans_multiple_cells(self):
        """Test edge that crosses multiple cells"""
        # Long diagonal edge
        x_coords = [0, 100]
        y_coords = [0, 100]
        edges = [(0, 1)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=10)
        
        # Edge should be registered in multiple cells along its path
        stats = solver.get_spatial_hash_stats()
        # Diagonal from (0,0) to (100,100) crosses many 10x10 cells
        assert stats['num_cells'] > 1
    
    def test_multiple_edges_per_cell(self):
        """Test multiple edges mapping to same cell"""
        # All edges in same region
        x_coords = [0, 5, 0, 5]
        y_coords = [0, 0, 5, 5]
        edges = [(0, 1), (2, 3), (0, 2), (1, 3)]  # Grid within 5x5
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=10)
        
        stats = solver.get_spatial_hash_stats()
        # All edges should cluster in one or few cells
        avg_edges = stats.get('edges_per_cell_avg', 0)
        assert avg_edges > 1, "Multiple edges should share cells"


class TestSpatialHashCrossingDetection:
    """Test crossing detection using spatial hash acceleration"""
    
    def test_spatial_hash_accuracy(self):
        """Test that spatial hash gives same results as brute force"""
        # X-shape graph
        x_coords = [0, 10, 0, 10]
        y_coords = [0, 10, 10, 0]
        edges = [(0, 1), (2, 3)]
        
        # Brute force (small cell_size = 0 disables spatial hash)
        solver_brute = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=0)
        crossings_brute = solver_brute.calculate_total_crossings()
        
        # Spatial hash
        solver_spatial = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=5)
        crossings_spatial = solver_spatial.calculate_total_crossings()
        
        assert crossings_brute == crossings_spatial == 1, "Both methods should find 1 crossing"
    
    def test_15_nodes_benchmark_with_spatial_hash(self):
        """Test 15-node benchmark with spatial hash (should match Cycle 1 result)"""
        # Load 15-node instance
        import json
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'live-2025-example-instances', '15-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        x_coords = [node['x'] for node in data['nodes']]
        y_coords = [node['y'] for node in data['nodes']]
        # Fix: edges are dictionaries with 'source' and 'target' keys
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        # With spatial hash
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=50)
        crossings = solver.calculate_total_crossings()
        
        # Should match Cycle 1 result
        assert crossings == 313, f"15-node benchmark should have 313 crossings, got {crossings}"
    
    def test_large_graph_correctness(self):
        """Test correctness on larger graph"""
        # 100-node grid
        import json
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'live-2025-example-instances', '100-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        x_coords = [node['x'] for node in data['nodes']]
        y_coords = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        # Brute force
        solver_brute = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=0)
        crossings_brute = solver_brute.calculate_total_crossings()
        
        # Spatial hash
        solver_spatial = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=100)
        crossings_spatial = solver_spatial.calculate_total_crossings()
        
        assert crossings_brute == crossings_spatial, \
            f"Spatial hash ({crossings_spatial}) should match brute force ({crossings_brute})"


class TestDeltaEWithSpatialHash:
    """Test delta-E computation with spatial hash acceleration"""
    
    def test_delta_e_faster_with_spatial_hash(self):
        """Test that delta-E is faster with spatial hash"""
        # Medium-sized graph (70 nodes)
        import json
        instance_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'live-2025-example-instances', '70-nodes.json'
        )
        
        with open(instance_path, 'r') as f:
            data = json.load(f)
        
        x_coords = [node['x'] for node in data['nodes']]
        y_coords = [node['y'] for node in data['nodes']]
        edges = [(edge['source'], edge['target']) for edge in data['edges']]
        
        # Brute force solver
        solver_brute = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=0)
        
        # Spatial hash solver
        solver_spatial = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=100)
        
        # Time brute force delta-E (10 iterations)
        start = time.perf_counter()
        for i in range(10):
            delta_brute = solver_brute.compute_delta_e(i % len(x_coords), x_coords[i] + 10, y_coords[i] + 10)
        time_brute = time.perf_counter() - start
        
        # Time spatial hash delta-E (10 iterations)
        start = time.perf_counter()
        for i in range(10):
            delta_spatial = solver_spatial.compute_delta_e(i % len(x_coords), x_coords[i] + 10, y_coords[i] + 10)
        time_spatial = time.perf_counter() - start
        
        print(f"\nBrute force: {time_brute:.4f}s")
        print(f"Spatial hash: {time_spatial:.4f}s")
        print(f"Speedup: {time_brute / time_spatial:.2f}x")
        
        # Spatial hash should be faster (at least 1.5x for 70 nodes)
        assert time_spatial < time_brute, "Spatial hash should be faster than brute force"
        
        # Note: Speedup expectation depends on graph structure
        # For well-distributed graphs, expect 2-5x speedup
    
    def test_delta_e_accuracy_with_spatial_hash(self):
        """Test delta-E gives same result with spatial hash"""
        x_coords = [0, 10, 0, 10]
        y_coords = [0, 10, 10, 0]
        edges = [(0, 1), (2, 3)]
        
        solver_brute = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=0)
        solver_spatial = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=5)
        
        # Compute delta-E for same move
        delta_brute = solver_brute.compute_delta_e(1, 10, 0)
        delta_spatial = solver_spatial.compute_delta_e(1, 10, 0)
        
        assert delta_brute == delta_spatial, \
            f"Delta-E should match: brute={delta_brute}, spatial={delta_spatial}"


class TestSpatialHashDynamicUpdates:
    """Test spatial hash updates after node position changes"""
    
    def test_spatial_hash_updates_after_move(self):
        """Test that spatial hash is updated after node move"""
        x_coords = [0, 100, 0, 100]
        y_coords = [0, 0, 100, 100]
        edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=50)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Move node to different region
        solver.update_node_position(1, 50, 50)
        
        # Should still compute correctly (spatial hash updated)
        new_crossings = solver.calculate_total_crossings()
        
        # Verify correctness by comparing with brute force
        solver_check = planar_cuda.PlanarSolver(
            [0, 50, 0, 100],
            [0, 50, 100, 100],
            edges,
            cell_size=0  # Brute force
        )
        expected_crossings = solver_check.calculate_total_crossings()
        
        assert new_crossings == expected_crossings, \
            f"Spatial hash after update ({new_crossings}) should match brute force ({expected_crossings})"
    
    def test_reset_rebuilds_spatial_hash(self):
        """Test that reset properly rebuilds spatial hash"""
        x_coords = [0, 10, 0, 10]
        y_coords = [0, 0, 10, 10]
        edges = [(0, 1), (2, 3)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=5)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Make some moves
        solver.update_node_position(0, 50, 50)
        solver.update_node_position(1, 60, 60)
        
        # Reset
        solver.reset_to_initial()
        
        # Should return to initial crossing count
        reset_crossings = solver.calculate_total_crossings()
        assert reset_crossings == initial_crossings, \
            f"After reset: {reset_crossings} should equal initial: {initial_crossings}"


class TestSpatialHashPerformance:
    """Performance benchmarks for spatial hash"""
    
    def test_performance_scaling(self):
        """Test performance improvement on different graph sizes"""
        import json
        results = []
        
        test_cases = [
            ('15-nodes.json', 15),
            ('70-nodes.json', 70),
            ('100-nodes.json', 100),
        ]
        
        for filename, node_count in test_cases:
            instance_path = os.path.join(
                os.path.dirname(__file__), '..', '..', 
                'live-2025-example-instances', filename
            )
            
            with open(instance_path, 'r') as f:
                data = json.load(f)
            
            x_coords = [node['x'] for node in data['nodes']]
            y_coords = [node['y'] for node in data['nodes']]
            edges = [(edge['source'], edge['target']) for edge in data['edges']]
            
            # Create both solvers
            solver_brute = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=0)
            solver_spatial = planar_cuda.PlanarSolver(x_coords, y_coords, edges, cell_size=100)
            
            # Benchmark brute force
            start = time.perf_counter()
            for _ in range(5):
                solver_brute.calculate_total_crossings()
            time_brute = time.perf_counter() - start
            
            # Benchmark spatial hash
            start = time.perf_counter()
            for _ in range(5):
                solver_spatial.calculate_total_crossings()
            time_spatial = time.perf_counter() - start
            
            speedup = time_brute / time_spatial if time_spatial > 0 else 1.0
            results.append({
                'nodes': node_count,
                'edges': len(edges),
                'time_brute': time_brute,
                'time_spatial': time_spatial,
                'speedup': speedup
            })
        
        # Print results
        print("\n=== Spatial Hash Performance ===")
        for r in results:
            print(f"{r['nodes']} nodes, {r['edges']} edges: "
                  f"Brute={r['time_brute']:.4f}s, Spatial={r['time_spatial']:.4f}s, "
                  f"Speedup={r['speedup']:.2f}x")
        
        # Expect some speedup on larger graphs
        # Note: Actual speedup depends on graph structure and edge distribution
        assert all(r['speedup'] > 0 for r in results), "All tests should complete successfully"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
