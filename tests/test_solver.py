import unittest
import numpy as np
from src.solver import SimulatedAnnealingSolver

class TestSolver(unittest.TestCase):
    def test_solver_moves_nodes(self):
        # Create a simple crossing: (0,0)-(2,2) and (0,2)-(2,0)
        nodes = [
            {'id': 0, 'x': 0.0, 'y': 0.0},
            {'id': 1, 'x': 2.0, 'y': 2.0},
            {'id': 2, 'x': 0.0, 'y': 2.0},
            {'id': 3, 'x': 2.0, 'y': 0.0}
        ]
        edges = [
            {'source': 0, 'target': 1},
            {'source': 2, 'target': 3}
        ]
        
        solver = SimulatedAnnealingSolver(nodes, edges, 10, 10)
        
        initial_x, initial_y = solver.current_state()
        initial_energy = solver.energy(initial_x, initial_y)
        
        # Run solver for a bit
        # High temp to force moves
        new_x, new_y, new_energy, new_temp = solver.solve(iterations=100, temp=5.0, cooling_rate=0.9)
        
        # Check if positions changed
        self.assertFalse(np.array_equal(initial_x, new_x) and np.array_equal(initial_y, new_y), "Nodes should have moved")
        
        # Check if temp decreased
        self.assertLess(new_temp, 5.0, "Temperature should decrease")

if __name__ == '__main__':
    unittest.main()
