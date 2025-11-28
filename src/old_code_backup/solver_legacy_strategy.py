"""
Legacy Solver Strategy - Wrapper for Original Implementation
Uses the existing SimulatedAnnealingSolver with numpy arrays.
"""
import json
import numpy as np
from typing import Dict, Any

from solver_strategy import ISolverStrategy, SolverFactory
from solver import SimulatedAnnealingSolver as LegacySolver


class LegacySolverStrategy(ISolverStrategy):
    """
    Wraps the original numpy-based solver implementation.
    Maintains backward compatibility with existing code.
    """
    
    def __init__(self):
        """Initialize legacy solver strategy."""
        self.solver = None
        self.nodes = []
        self.edges = []
        self.width = 0
        self.height = 0
    
    def load_from_json(self, json_path: str):
        """Load graph from JSON using legacy format."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.nodes = data['nodes']
        self.edges = data['edges']
        self.width = data.get('width', 1000000)
        self.height = data.get('height', 1000000)
        
        # Create legacy solver
        self.solver = LegacySolver(
            self.nodes, 
            self.edges, 
            self.width, 
            self.height
        )
    
    def solve(self, iterations: int = 1000, **kwargs) -> Dict[str, Any]:
        """
        Run legacy solver.
        
        Args:
            iterations: Number of iterations
            **kwargs: Can include 'temp', 'cooling_rate'
        """
        if self.solver is None:
            raise RuntimeError("Must load data first using load_from_json()")
        
        # Extract parameters
        temp = kwargs.get('temp', 10.0)
        cooling_rate = kwargs.get('cooling_rate', 0.995)
        
        # Run solver
        best_x, best_y, final_energy, final_temp = self.solver.solve(
            iterations=iterations,
            temp=temp,
            cooling_rate=cooling_rate
        )
        
        # Update nodes with final positions
        for i, node in enumerate(self.nodes):
            node['x'] = float(best_x[i])
            node['y'] = float(best_y[i])
        
        # Calculate final statistics
        k = np.max(self.solver.edge_crossings) if len(self.solver.edge_crossings) > 0 else 0
        total = np.sum(self.solver.edge_crossings) // 2
        
        return {
            'nodes': self.nodes,
            'stats': {
                'iterations': iterations,
                'final_temp': final_temp,
                'method': 'legacy_numpy'
            },
            'energy': final_energy,
            'k': int(k),
            'total_crossings': int(total)
        }
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics from legacy solver."""
        if self.solver is None:
            return {'error': 'No solver initialized'}
        
        energy = self.solver.energy()
        k = np.max(self.solver.edge_crossings) if len(self.solver.edge_crossings) > 0 else 0
        total = np.sum(self.solver.edge_crossings) // 2
        
        return {
            'energy': energy,
            'k': int(k),
            'total_crossings': int(total),
            'num_nodes': len(self.nodes),
            'num_edges': len(self.edges)
        }
    
    def export_to_json(self, output_path: str):
        """Export current solution to JSON."""
        data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'width': self.width,
            'height': self.height
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)


# Register with factory
SolverFactory.register_strategy('legacy', LegacySolverStrategy)
