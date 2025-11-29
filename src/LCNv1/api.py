"""
LCNv1 公共 API
提供統一的求解器接口
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .strategies import ISolverStrategy, SolverFactory, StrategyType


@dataclass
class OptimizationResult:
    """
    優化結果數據類
    
    Attributes:
        energy: 最終能量值
        k: 最大交叉數 (單條邊的最大交叉數)
        total_crossings: 總交叉數
        iterations: 實際迭代次數
        acceptance_rate: 接受率
        time: 運行時間 (秒)
        initial_energy: 初始能量值
        initial_k: 初始 K 值
        initial_crossings: 初始交叉數
        improvement: 改進百分比
    """
    energy: float
    k: int
    total_crossings: int
    iterations: int
    nodes: list = None  # Added to support returning positions
    edge_crossings: list = None # Added to support red edge indicators
    acceptance_rate: float = 0.0
    time: float = 0.0
    initial_energy: float = 0.0
    initial_k: int = 0
    initial_crossings: int = 0
    
    @property
    def improvement(self) -> float:
        """計算改進百分比"""
        if self.initial_energy == 0:
            return 0.0
        return (self.initial_energy - self.energy) / self.initial_energy * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典"""
        return {
            'energy': self.energy,
            'k': self.k,
            'total_crossings': self.total_crossings,
            'iterations': self.iterations,
            'acceptance_rate': self.acceptance_rate,
            'time': self.time,
            'initial_energy': self.initial_energy,
            'initial_k': self.initial_k,
            'initial_crossings': self.initial_crossings,
            'improvement': self.improvement,
        }


class LCNSolver:
    """
    LCN 求解器 - 統一接口
    
    這是 LCNv1 系統的主要接口，提供圖形佈局優化功能。
    
    Examples:
        基本使用:
        >>> solver = LCNSolver()
        >>> solver.load_from_json('input.json')
        >>> result = solver.optimize(iterations=1000)
        >>> solver.export_to_json('output.json')
        
        指定策略:
        >>> solver = LCNSolver(strategy='numba')
        >>> result = solver.optimize(iterations=1000)
        
        自定義參數:
        >>> solver = LCNSolver(
        ...     strategy='new',
        ...     w_cross=100.0,
        ...     w_len=1.0,
        ...     power=2
        ... )
    """
    
    def __init__(
        self,
        strategy: str = 'numba',
        w_cross: float = 100.0,
        w_len: float = 1.0,
        power: int = 2,
        **kwargs
    ):
        """
        初始化求解器
        
        Args:
            strategy: 求解策略 ('legacy', 'new', 'numba')
            w_cross: 交叉懲罰權重
            w_len: 邊長懲罰權重
            power: 交叉懲罰指數
            **kwargs: 傳遞給策略的其他參數
        """
        # 確保策略已註冊
        self._register_strategies()
        
        # 創建策略實例
        self._strategy: ISolverStrategy = SolverFactory.create_solver(
            strategy,
            w_cross=w_cross,
            w_len=w_len,
            power=power,
            **kwargs
        )
        
        self._strategy_name = strategy
        self._initial_stats: Optional[Dict] = None
    
    def _register_strategies(self):
        """註冊所有可用策略"""
        try:
            # 導入並註冊策略
            from .strategies import legacy, new, numba_jit
        except ImportError as e:
            print(f"Warning: Some strategies failed to load: {e}")
    
    def load_from_json(self, json_path: str):
        """
        從 JSON 文件加載圖形
        
        JSON 格式:
        {
            "nodes": [
                {"id": 0, "x": 100, "y": 200},
                ...
            ],
            "edges": [
                {"source": 0, "target": 1},
                ...
            ]
        }
        
        Args:
            json_path: JSON 文件路徑
        """
        self._strategy.load_from_json(json_path)
        self._initial_stats = self._strategy.get_current_stats()
    
    def optimize(
        self,
        iterations: int = 1000,
        initial_temp: float = 50.0,
        cooling_rate: float = 0.995,
        reheat_threshold: int = 500,
        **kwargs
    ) -> OptimizationResult:
        """
        執行優化
        
        Args:
            iterations: 迭代次數
            initial_temp: 初始溫度 (模擬退火)
            cooling_rate: 降溫率
            reheat_threshold: 重新加熱閾值
            **kwargs: 傳遞給策略的其他參數
        
        Returns:
            OptimizationResult: 優化結果
        """
        result = self._strategy.solve(
            iterations=iterations,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            reheat_threshold=reheat_threshold,
            **kwargs
        )
        
        # 構建結果對象
        # 構建結果對象
        return OptimizationResult(
            energy=result['energy'],
            k=result['k'],
            total_crossings=result['total_crossings'],
            iterations=result.get('iterations', iterations),
            nodes=result.get('nodes'),  # Pass updated nodes
            edge_crossings=result.get('edge_crossings'), # Pass edge crossings
            acceptance_rate=result.get('acceptance_rate', 0.0),
            time=result.get('time', 0.0),
            initial_energy=self._initial_stats['energy'] if self._initial_stats else 0.0,
            initial_k=self._initial_stats['k'] if self._initial_stats else 0,
            initial_crossings=self._initial_stats['total_crossings'] if self._initial_stats else 0,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        獲取當前狀態統計信息
        
        Returns:
            包含 energy, k, total_crossings 的字典
        """
        return self._strategy.get_current_stats()
    
    def export_to_json(self, output_path: str):
        """
        導出優化結果到 JSON 文件
        
        Args:
            output_path: 輸出文件路徑
        """
        self._strategy.export_to_json(output_path)
    
    @property
    def strategy_name(self) -> str:
        """當前使用的策略名稱"""
        return self._strategy_name
    
    @staticmethod
    def list_strategies() -> list:
        """
        列出所有可用策略
        
        Returns:
            策略名稱列表
        """
        return SolverFactory.list_strategies()
    
    def __repr__(self):
        return f"LCNSolver(strategy='{self._strategy_name}')"


__all__ = ['LCNSolver', 'OptimizationResult', 'StrategyType']
