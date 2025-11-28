"""
Solver Strategy Pattern - Abstract Interface
Allows switching between different solver implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import json


class ISolverStrategy(ABC):
    """
    Abstract interface for solver strategies.
    Different implementations can use different architectures.
    """
    
    @abstractmethod
    def load_from_json(self, json_path: str):
        """
        Load graph data from JSON file.
        
        Args:
            json_path: Path to JSON file
        """
        pass
    
    @abstractmethod
    def solve(self, iterations: int = 1000, **kwargs) -> Dict[str, Any]:
        """
        Run the optimization algorithm.
        
        Args:
            iterations: Number of iterations to run
            **kwargs: Strategy-specific parameters
            
        Returns:
            Dictionary containing:
            - 'nodes': Final node positions
            - 'stats': Optimization statistics
            - 'energy': Final energy value
            - 'k': Final maximum crossings per edge
            - 'total_crossings': Final total crossings
        """
        pass
    
    @abstractmethod
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current state statistics.
        
        Returns:
            Dictionary with current metrics
        """
        pass
    
    @abstractmethod
    def export_to_json(self, output_path: str):
        """
        Export current solution to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        pass


class SolverFactory:
    """
    Factory for creating solver strategies.
    """
    
    _strategies = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """Register a solver strategy."""
        cls._strategies[name] = strategy_class
    
    @classmethod
    def create_solver(cls, strategy_name: str, **kwargs) -> ISolverStrategy:
        """
        Create a solver instance.
        
        Args:
            strategy_name: Name of the strategy ('legacy' or 'new')
            **kwargs: Strategy-specific initialization parameters
            
        Returns:
            Solver instance
        """
        if strategy_name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            )
        
        return cls._strategies[strategy_name](**kwargs)
    
    @classmethod
    def list_strategies(cls):
        """List all available strategies."""
        return list(cls._strategies.keys())
