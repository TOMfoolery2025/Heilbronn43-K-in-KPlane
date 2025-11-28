"""
LCNv1 求解策略模塊
包含所有優化策略的實現
"""

from enum import Enum
from .base import ISolverStrategy, SolverFactory


class StrategyType(str, Enum):
    """求解策略類型枚舉"""
    LEGACY = 'legacy'
    NEW = 'new'
    NUMBA = 'numba'
    CUDA = 'cuda'  # 預留


# 自動註冊策略
from . import register

__all__ = [
    'ISolverStrategy',
    'SolverFactory',
    'StrategyType',
]
