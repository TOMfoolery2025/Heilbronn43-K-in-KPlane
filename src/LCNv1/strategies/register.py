"""
自動註冊所有策略
在導入時執行
"""

# 導入基礎接口
from .base import ISolverStrategy, SolverFactory

# 導入並註冊 Legacy 策略
try:
    import sys
    from pathlib import Path
    
    # 添加 src 到路徑以訪問原始 solver.py
    src_path = Path(__file__).parent.parent.parent
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    from .legacy import LegacySolverStrategy
    SolverFactory.register_strategy('legacy', LegacySolverStrategy)
    print("[OK] Legacy strategy registered")
except ImportError as e:
    print(f"[WARN] Legacy strategy registration failed: {e}")

# 導入並註冊 New 策略
try:
    from .new import NewArchitectureSolverStrategy
    SolverFactory.register_strategy('new', NewArchitectureSolverStrategy)
    print("[OK] New strategy registered")
except ImportError as e:
    print(f"[WARN] New strategy registration failed: {e}")

# 導入並註冊 Numba 策略
try:
    from .numba_jit import NumbaSolverStrategy
    SolverFactory.register_strategy('numba', NumbaSolverStrategy)
    print("[OK] Numba strategy registered")
except ImportError as e:
    print(f"[WARN] Numba strategy registration failed: {e}")

# 導入並註冊 CUDA 策略 (如果可用)
try:
    import sys
    cuda_strategy_path = Path(__file__).parent.parent.parent
    if str(cuda_strategy_path) not in sys.path:
        sys.path.insert(0, str(cuda_strategy_path))
    
    from solver_cuda_strategy import CUDASolverStrategy
    SolverFactory.register_strategy('cuda', CUDASolverStrategy)
    print("[OK] CUDA strategy registered")
except ImportError:
    pass  # CUDA 策略是可選的

__all__ = ['ISolverStrategy', 'SolverFactory', 'StrategyType']
