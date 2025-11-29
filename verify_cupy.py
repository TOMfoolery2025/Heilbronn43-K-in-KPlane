import cupy as cp
import numpy as np

try:
    print(f"CuPy Version: {cp.__version__}")
    print(f"CUDA Device: {cp.cuda.Device(0).compute_capability}")
    
    # Simple test
    x_gpu = cp.array([1, 2, 3])
    y_gpu = cp.array([4, 5, 6])
    z_gpu = x_gpu + y_gpu
    print(f"Test Calculation: {z_gpu}")
    print("CuPy is working correctly!")
except Exception as e:
    print(f"CuPy Error: {e}")
