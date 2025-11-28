"""
Cycle 2: State Management Tests
================================
Test suite for GPU-resident state updates and incremental crossing calculations.

Test-Driven Development (TDD) Approach:
1. RED: Write failing tests first
2. GREEN: Implement minimal code to pass
3. REFACTOR: Optimize while keeping tests green

Test Coverage:
- Node position updates in GPU memory
- Delta-E computation (incremental crossing changes)
- State consistency across multiple operations
- Memory persistence validation
"""

import pytest
import sys
import os

# Add CUDA DLL path and module path
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build_artifacts'))

import planar_cuda


class TestNodePositionUpdate:
    """Test GPU-resident node position updates"""
    
    def test_single_node_update(self):
        """Test updating a single node's position"""
        # Create an X-shape that becomes planar after update
        # Initial: (0,0)-(10,10) crosses (0,10)-(10,0) = 1 crossing
        x_coords = [0, 10, 0, 10]
        y_coords = [0, 10, 10, 0]
        edges = [(0, 1), (2, 3)]  # Two crossing diagonals
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Initial state should have 1 crossing
        initial_crossings = solver.calculate_total_crossings()
        assert initial_crossings == 1, "X-shape should have 1 crossing initially"
        
        # Update node 1 to align with node 3, making edges parallel
        # Move from (10,10) to (10,0)
        solver.update_node_position(1, 10, 0)
        
        # Verify the update - now edges don't cross (both vertical/degenerate)
        new_crossings = solver.calculate_total_crossings()
        assert new_crossings == 0, "After update, should have 0 crossings"
    
    def test_multiple_updates_same_node(self):
        """Test multiple updates to the same node"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Update 1: Move node 2 to (15, 5)
        solver.update_node_position(2, 15, 5)
        crossings_1 = solver.calculate_total_crossings()
        
        # Update 2: Move node 2 to (5, -2)
        solver.update_node_position(2, 5, -2)
        crossings_2 = solver.calculate_total_crossings()
        
        # Update 3: Move back to original position
        solver.update_node_position(2, 5, 10)
        crossings_3 = solver.calculate_total_crossings()
        
        assert crossings_3 == 0, "Should return to planar after restoring original position"
    
    def test_update_multiple_nodes(self):
        """Test updating different nodes"""
        # Square graph
        x_coords = [0, 10, 10, 0]
        y_coords = [0, 0, 10, 10]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Planar square
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        assert solver.calculate_total_crossings() == 0, "Square should be planar"
        
        # Update node 0
        solver.update_node_position(0, 5, 5)
        
        # Update node 2
        solver.update_node_position(2, 5, 5)
        
        # Now nodes 0 and 2 are at same position - degenerate but no crossings
        crossings = solver.calculate_total_crossings()
        assert crossings >= 0, "Should handle degenerate case"
    
    def test_get_current_position(self):
        """Test retrieving current node positions from GPU"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Get initial positions
        x0, y0 = solver.get_node_position(0)
        assert (x0, y0) == (0, 0), "Node 0 should be at (0,0)"
        
        x2, y2 = solver.get_node_position(2)
        assert (x2, y2) == (5, 10), "Node 2 should be at (5,10)"
        
        # Update and verify
        solver.update_node_position(2, 20, 30)
        x2_new, y2_new = solver.get_node_position(2)
        assert (x2_new, y2_new) == (20, 30), "Node 2 should be at new position (20,30)"


class TestDeltaEComputation:
    """Test incremental energy (crossing count) changes"""
    
    def test_delta_e_simple_move(self):
        """Test delta-E computation for a simple move"""
        # X-shape graph with 1 crossing
        x_coords = [0, 10, 0, 10]
        y_coords = [0, 10, 10, 0]
        edges = [(0, 1), (2, 3)]  # Two crossing edges
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Current crossings
        current_crossings = solver.calculate_total_crossings()
        assert current_crossings == 1, "X-shape should have 1 crossing"
        
        # Compute delta-E for moving node 1 to (10, 0)
        # This would align nodes 1 and 3, removing the crossing
        delta_e = solver.compute_delta_e(node_id=1, new_x=10, new_y=0)
        
        # Delta-E should be -1 (removing 1 crossing)
        assert delta_e == -1, f"Moving node 1 to (10,0) should reduce crossings by 1, got delta_e={delta_e}"
        
        # Verify the actual move produces the same result
        solver.update_node_position(1, 10, 0)
        new_crossings = solver.calculate_total_crossings()
        assert new_crossings == 0, "After move, should have 0 crossings"
        assert new_crossings == current_crossings + delta_e, "Delta-E prediction should match actual change"
    
    def test_delta_e_without_update(self):
        """Test that compute_delta_e doesn't modify state"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Compute delta-E for a hypothetical move
        delta_e = solver.compute_delta_e(node_id=2, new_x=5, new_y=-2)
        
        # Verify state unchanged
        crossings_after_delta = solver.calculate_total_crossings()
        assert crossings_after_delta == initial_crossings, "compute_delta_e should not modify state"
        
        # Verify position unchanged
        x2, y2 = solver.get_node_position(2)
        assert (x2, y2) == (5, 10), "Node position should be unchanged after delta-E computation"
    
    def test_delta_e_multiple_edges(self):
        """Test delta-E with multiple edges affected"""
        # Star graph: center node connected to 4 outer nodes
        x_coords = [5, 0, 10, 10, 0]  # Center, Top, Right, Bottom, Left
        y_coords = [5, 0, 5, 10, 5]
        edges = [
            (0, 1),  # Center to Top
            (0, 2),  # Center to Right
            (0, 3),  # Center to Bottom
            (0, 4),  # Center to Left
        ]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Move center node to create/remove crossings
        delta_e = solver.compute_delta_e(node_id=0, new_x=0, new_y=0)
        
        # Apply the move and verify
        solver.update_node_position(0, 0, 0)
        new_crossings = solver.calculate_total_crossings()
        
        actual_delta = new_crossings - initial_crossings
        assert delta_e == actual_delta, f"Delta-E prediction ({delta_e}) should match actual change ({actual_delta})"


class TestStateConsistency:
    """Test state consistency and memory management"""
    
    def test_consistency_after_many_updates(self):
        """Test state remains consistent after many updates"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Perform 100 random-ish updates
        import random
        random.seed(42)
        
        for i in range(100):
            node_id = i % 3
            new_x = random.randint(-20, 20)
            new_y = random.randint(-20, 20)
            solver.update_node_position(node_id, new_x, new_y)
        
        # Should still be able to compute crossings
        crossings = solver.calculate_total_crossings()
        assert crossings >= 0, "Should successfully compute crossings after many updates"
        
        # Verify positions can be retrieved
        for node_id in range(3):
            x, y = solver.get_node_position(node_id)
            assert isinstance(x, int) and isinstance(y, int), "Should retrieve valid positions"
    
    def test_reset_to_initial_state(self):
        """Test resetting graph to initial configuration"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        initial_crossings = solver.calculate_total_crossings()
        
        # Make some updates
        solver.update_node_position(0, 5, 5)
        solver.update_node_position(1, 15, 15)
        solver.update_node_position(2, 10, 20)
        
        # Reset to initial state
        solver.reset_to_initial()
        
        # Verify back to initial configuration
        crossings = solver.calculate_total_crossings()
        assert crossings == initial_crossings, "Should return to initial crossing count"
        
        x0, y0 = solver.get_node_position(0)
        assert (x0, y0) == (0, 0), "Node 0 should be back at (0,0)"
        
        x2, y2 = solver.get_node_position(2)
        assert (x2, y2) == (5, 10), "Node 2 should be back at (5,10)"


class TestBoundaryConditions:
    """Test edge cases and boundary conditions"""
    
    def test_update_invalid_node_id(self):
        """Test updating non-existent node"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Try to update node 10 (doesn't exist)
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            solver.update_node_position(10, 0, 0)
    
    def test_update_negative_node_id(self):
        """Test updating with negative node ID"""
        x_coords = [0, 10, 5]
        y_coords = [0, 0, 10]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        with pytest.raises((IndexError, RuntimeError, ValueError)):
            solver.update_node_position(-1, 0, 0)
    
    def test_large_coordinate_values(self):
        """Test with very large coordinate values"""
        x_coords = [0, 1000000, 500000]
        y_coords = [0, 0, 1000000]
        edges = [(0, 1), (1, 2), (2, 0)]
        
        solver = planar_cuda.PlanarSolver(x_coords, y_coords, edges)
        
        # Update to even larger values
        solver.update_node_position(2, 2000000, 3000000)
        
        # Should handle large values without overflow
        crossings = solver.calculate_total_crossings()
        assert crossings >= 0, "Should handle large coordinates"
        
        x2, y2 = solver.get_node_position(2)
        assert (x2, y2) == (2000000, 3000000), "Should correctly store large coordinates"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
