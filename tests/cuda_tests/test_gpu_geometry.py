"""
Test-Driven Development: Cycle 1 - Geometry Verification
Tests the CUDA implementation against Python ground truth.

Following TDD Cycle:
1. RED: Write failing tests (this file)
2. GREEN: Implement CUDA code to pass tests
3. REFACTOR: Optimize while keeping tests green
"""
import pytest
import sys
import os
import json
from pathlib import Path

# Setup paths for module import
project_root = Path(__file__).parent.parent.parent
build_artifacts = project_root / "build_artifacts"
sys.path.insert(0, str(build_artifacts))

# Add CUDA DLL directory (Windows)
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
if os.path.exists(cuda_path):
    os.add_dll_directory(cuda_path)


class TestCycle1GeometryVerification:
    """
    OOA/OOD Test Suite: Cycle 1 - Geometry on GPU
    
    Entity Under Test: PlanarSolver class
    Behavior Under Test: calculate_total_crossings()
    
    Success Criteria:
    - GPU crossing count matches Python ground truth
    - Handles edge cases (empty graph, single edge, no crossings)
    - Memory management (no leaks, correct cleanup)
    """
    
    @pytest.fixture(scope="class")
    def planar_cuda_module(self):
        """Fixture to import and validate the CUDA module."""
        try:
            import planar_cuda
            return planar_cuda
        except ImportError as e:
            pytest.skip(f"planar_cuda module not built: {e}")
    
    @pytest.fixture
    def sample_triangle(self):
        """
        Simple triangle graph for basic validation.
        
        Topology:
            (0,10)
              2
             /|\\
            / | \\
           /  |  \\
          /   |   \\
        (0,0) 0---1 (10,0)
        
        3 nodes, 3 edges, 0 crossings (planar)
        """
        return {
            'nodes_x': [0, 10, 0],
            'nodes_y': [0, 0, 10],
            'edges': [(0, 1), (1, 2), (2, 0)]
        }
    
    @pytest.fixture
    def sample_cross(self):
        """
        Simple X-shaped graph with exactly 1 crossing.
        
        Topology:
            (0,10)      (10,10)
              0-----------1
               \\         /
                \\       /
                 \\     /    <- crossing point (5,5)
                  \\   /
                   \\ /
                    X
                   / \\
                  /   \\
                 /     \\
                /       \\
            (0,0)       (10,0)
              2-----------3
        
        4 nodes, 2 edges, 1 crossing
        """
        return {
            'nodes_x': [0, 10, 0, 10],
            'nodes_y': [10, 10, 0, 0],
            'edges': [(0, 3), (1, 2)]  # Two diagonals that cross
        }
    
    @pytest.fixture
    def load_15_nodes(self):
        """Load the 15-nodes benchmark from live examples."""
        json_path = project_root / "live-2025-example-instances" / "15-nodes.json"
        with open(json_path) as f:
            data = json.load(f)
        
        nodes_x = [n['x'] for n in data['nodes']]
        nodes_y = [n['y'] for n in data['nodes']]
        edges = [(e['source'], e['target']) for e in data['edges']]
        
        return {
            'nodes_x': nodes_x,
            'nodes_y': nodes_y,
            'edges': edges
        }
    
    def calculate_crossings_python(self, nodes_x, nodes_y, edges):
        """
        Python reference implementation using the verified geometry core.
        This is our ground truth.
        """
        from src.LCNv1.core.geometry import Point, GeometryCore
        
        # Convert to Point objects
        points = [Point(x, y) for x, y in zip(nodes_x, nodes_y)]
        
        # Count crossings using O(E^2) brute force
        crossing_count = 0
        num_edges = len(edges)
        
        for i in range(num_edges):
            u1, v1 = edges[i]
            p1, p2 = points[u1], points[v1]
            
            for j in range(i + 1, num_edges):
                u2, v2 = edges[j]
                q1, q2 = points[u2], points[v2]
                
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    crossing_count += 1
        
        return crossing_count
    
    def test_module_import(self, planar_cuda_module):
        """Test 1: Verify module can be imported and has correct interface."""
        assert hasattr(planar_cuda_module, 'PlanarSolver')
        assert hasattr(planar_cuda_module, '__version__')
        print(f"✓ Module version: {planar_cuda_module.__version__}")
    
    def test_empty_graph(self, planar_cuda_module):
        """Test 2: Edge case - empty graph should have 0 crossings."""
        solver = planar_cuda_module.PlanarSolver([], [], [])
        assert solver.calculate_total_crossings() == 0
        print("✓ Empty graph handled correctly")
    
    def test_single_edge(self, planar_cuda_module):
        """Test 3: Single edge should have 0 crossings."""
        solver = planar_cuda_module.PlanarSolver([0, 10], [0, 0], [(0, 1)])
        assert solver.calculate_total_crossings() == 0
        print("✓ Single edge handled correctly")
    
    def test_triangle_planar(self, planar_cuda_module, sample_triangle):
        """Test 4: Triangle (planar graph) should have 0 crossings."""
        solver = planar_cuda_module.PlanarSolver(
            sample_triangle['nodes_x'],
            sample_triangle['nodes_y'],
            sample_triangle['edges']
        )
        gpu_crossings = solver.calculate_total_crossings()
        
        # Verify against Python
        python_crossings = self.calculate_crossings_python(
            sample_triangle['nodes_x'],
            sample_triangle['nodes_y'],
            sample_triangle['edges']
        )
        
        assert gpu_crossings == python_crossings == 0
        print(f"✓ Triangle: GPU={gpu_crossings}, Python={python_crossings}")
    
    def test_simple_crossing(self, planar_cuda_module, sample_cross):
        """Test 5: X-shaped graph should have exactly 1 crossing."""
        solver = planar_cuda_module.PlanarSolver(
            sample_cross['nodes_x'],
            sample_cross['nodes_y'],
            sample_cross['edges']
        )
        gpu_crossings = solver.calculate_total_crossings()
        
        # Verify against Python
        python_crossings = self.calculate_crossings_python(
            sample_cross['nodes_x'],
            sample_cross['nodes_y'],
            sample_cross['edges']
        )
        
        assert gpu_crossings == python_crossings == 1
        print(f"✓ Simple cross: GPU={gpu_crossings}, Python={python_crossings}")
    
    def test_15_nodes_benchmark(self, planar_cuda_module, load_15_nodes):
        """
        Test 6: CRITICAL TEST - 15-nodes benchmark
        
        This is the main verification test. The GPU implementation must
        produce EXACTLY the same crossing count as the verified Python version.
        """
        data = load_15_nodes
        
        # Create solver
        solver = planar_cuda_module.PlanarSolver(
            data['nodes_x'],
            data['nodes_y'],
            data['edges']
        )
        
        # Get GPU result
        gpu_crossings = solver.calculate_total_crossings()
        
        # Get Python ground truth
        python_crossings = self.calculate_crossings_python(
            data['nodes_x'],
            data['nodes_y'],
            data['edges']
        )
        
        # CRITICAL ASSERTION
        assert gpu_crossings == python_crossings, \
            f"GPU-Python mismatch! GPU={gpu_crossings}, Python={python_crossings}"
        
        print(f"✓ 15-nodes benchmark: GPU={gpu_crossings}, Python={python_crossings}")
        print(f"  Nodes: {len(data['nodes_x'])}, Edges: {len(data['edges'])}")
    
    def test_shared_endpoint_not_crossing(self, planar_cuda_module):
        """
        Test 7: Edges sharing endpoints should NOT count as crossings.
        
        Graph: 0---1---2 (chain)
        Edges (0,1) and (1,2) share node 1, not a crossing.
        """
        nodes_x = [0, 5, 10]
        nodes_y = [0, 0, 0]
        edges = [(0, 1), (1, 2)]
        
        solver = planar_cuda_module.PlanarSolver(nodes_x, nodes_y, edges)
        gpu_crossings = solver.calculate_total_crossings()
        
        python_crossings = self.calculate_crossings_python(nodes_x, nodes_y, edges)
        
        assert gpu_crossings == python_crossings == 0
        print("✓ Shared endpoint correctly ignored")
    
    def test_parallel_segments_not_crossing(self, planar_cuda_module):
        """
        Test 8: Parallel segments should NOT count as crossings.
        
        Graph:
            0---1
            |   |  (two parallel horizontal edges)
            2---3
        """
        nodes_x = [0, 10, 0, 10]
        nodes_y = [10, 10, 0, 0]
        edges = [(0, 1), (2, 3)]  # Two parallel horizontal segments
        
        solver = planar_cuda_module.PlanarSolver(nodes_x, nodes_y, edges)
        gpu_crossings = solver.calculate_total_crossings()
        
        python_crossings = self.calculate_crossings_python(nodes_x, nodes_y, edges)
        
        assert gpu_crossings == python_crossings == 0
        print("✓ Parallel segments correctly ignored")
    
    def test_collinear_segments_not_crossing(self, planar_cuda_module):
        """
        Test 9: Collinear overlapping segments should NOT count as crossings.
        
        Graph: 0---1---2---3 (all on same horizontal line)
        Edges (0,2) and (1,3) overlap but are collinear.
        """
        nodes_x = [0, 5, 10, 15]
        nodes_y = [0, 0, 0, 0]
        edges = [(0, 2), (1, 3)]  # Overlapping collinear segments
        
        solver = planar_cuda_module.PlanarSolver(nodes_x, nodes_y, edges)
        gpu_crossings = solver.calculate_total_crossings()
        
        python_crossings = self.calculate_crossings_python(nodes_x, nodes_y, edges)
        
        assert gpu_crossings == python_crossings == 0
        print("✓ Collinear segments correctly ignored")


class TestCycle1MemoryManagement:
    """
    Test memory allocation and cleanup.
    Ensures no CUDA memory leaks.
    """
    
    @pytest.fixture(scope="class")
    def planar_cuda_module(self):
        try:
            import planar_cuda
            return planar_cuda
        except ImportError as e:
            pytest.skip(f"planar_cuda module not built: {e}")
    
    def test_multiple_instances(self, planar_cuda_module):
        """Test 10: Create and destroy multiple solver instances."""
        for i in range(10):
            solver = planar_cuda_module.PlanarSolver(
                [0, 10, 5],
                [0, 0, 10],
                [(0, 1), (1, 2), (2, 0)]
            )
            crossings = solver.calculate_total_crossings()
            assert crossings == 0
            del solver  # Force cleanup
        
        print("✓ Multiple instances created/destroyed successfully")
    
    def test_large_graph(self, planar_cuda_module):
        """Test 11: Stress test with larger graph."""
        # Create a 100-node grid
        n = 10  # 10x10 grid
        nodes_x = [i % n for i in range(n * n)]
        nodes_y = [i // n for i in range(n * n)]
        
        # Create horizontal edges
        edges = []
        for i in range(n):
            for j in range(n - 1):
                node_id = i * n + j
                edges.append((node_id, node_id + 1))
        
        # Create vertical edges
        for i in range(n - 1):
            for j in range(n):
                node_id = i * n + j
                edges.append((node_id, node_id + n))
        
        solver = planar_cuda_module.PlanarSolver(nodes_x, nodes_y, edges)
        crossings = solver.calculate_total_crossings()
        
        # Grid graph is planar, should have 0 crossings
        assert crossings == 0
        print(f"✓ Large graph (100 nodes, {len(edges)} edges): {crossings} crossings")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
