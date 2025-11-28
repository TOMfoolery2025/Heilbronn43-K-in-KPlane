"""
Phase 1 Test: Verify Python -> C++ -> CUDA pipeline

This test ensures:
1. Python can import the CUDA module
2. Data can be transferred to GPU
3. CUDA kernel executes correctly
4. Results return to Python

Run: pytest tests/cuda_tests/test_phase1_pipeline.py -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add CUDA DLL directory
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')

# Add project root and build_artifacts to path
project_root = Path(__file__).parent.parent.parent
build_artifacts = project_root / "build_artifacts"
sys.path.insert(0, str(build_artifacts))
sys.path.insert(0, str(project_root))


def test_module_import():
    """Test 1: Can we import the CUDA module?"""
    try:
        import planar_cuda
        assert planar_cuda.cuda_enabled == True
        print(f"✅ Module imported successfully (version {planar_cuda.__version__})")
    except ImportError as e:
        pytest.fail(f"Failed to import planar_cuda: {e}\n"
                   f"Make sure you've built the module with CMake")


def test_simple_addition():
    """Test 2: Basic vector addition on GPU"""
    import planar_cuda
    
    a = [1, 2, 3, 4, 5]
    b = [10, 20, 30, 40, 50]
    expected = [11, 22, 33, 44, 55]
    
    result = planar_cuda.add_vectors(a, b)
    
    assert result == expected, f"Expected {expected}, got {result}"
    print(f"✅ Vector addition: {a} + {b} = {result}")


def test_large_vectors():
    """Test 3: GPU handles larger data"""
    import planar_cuda
    
    n = 10000
    a = list(range(n))
    b = list(range(n, 2*n))
    
    result = planar_cuda.add_vectors(a, b)
    
    # Verify first, middle, last elements
    assert result[0] == a[0] + b[0]
    assert result[n//2] == a[n//2] + b[n//2]
    assert result[-1] == a[-1] + b[-1]
    assert len(result) == n
    
    print(f"✅ Large vector test passed ({n} elements)")


def test_error_handling():
    """Test 4: Proper error handling for mismatched sizes"""
    import planar_cuda
    
    a = [1, 2, 3]
    b = [1, 2]  # Different size
    
    with pytest.raises(RuntimeError, match="Vector sizes must match"):
        planar_cuda.add_vectors(a, b)
    
    print("✅ Error handling works correctly")


def test_empty_vectors():
    """Test 5: Edge case - empty vectors"""
    import planar_cuda
    
    a = []
    b = []
    
    result = planar_cuda.add_vectors(a, b)
    assert result == []
    
    print("✅ Empty vector handling works")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Phase 1: Python ↔ C++ ↔ CUDA Pipeline Test")
    print("="*60 + "\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
