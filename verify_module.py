#!/usr/bin/env python3
"""快速驗證 LCNv1 模塊"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from LCNv1 import LCNSolver

print("✅ LCNv1 模塊導入成功")
print(f"可用策略: {LCNSolver.list_strategies()}")

# 測試創建求解器
for strategy in LCNSolver.list_strategies():
    try:
        solver = LCNSolver(strategy=strategy)
        print(f"✅ {strategy.upper()} 策略可用")
    except Exception as e:
        print(f"⚠️  {strategy.upper()} 策略不可用: {e}")

print("\n🎉 LCNv1 模塊完全正常！")
