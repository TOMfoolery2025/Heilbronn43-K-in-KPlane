# Hybrid Python + CUDA Development Roadmap

**Project**: LCN Solver GPU Acceleration  
**Approach**: Test-Driven Development (TDD)  
**Status**: Phase 1 - Environment Setup

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   Python Layer (Frontend/Control)      │
│   - JSON I/O                            │
│   - pytest testing                      │
│   - Visualization                       │
└──────────────┬──────────────────────────┘
               │ pybind11
┌──────────────▼──────────────────────────┐
│   C++ Binding Layer                     │
│   - Memory management                   │
│   - Python/GPU bridge                   │
└──────────────┬──────────────────────────┘
               │ CUDA API
┌──────────────▼──────────────────────────┐
│   CUDA Layer (Compute Kernel)           │
│   - Geometry kernels (__device__)       │
│   - Spatial hashing (__global__)        │
│   - SA solver loop                      │
└─────────────────────────────────────────┘
```

---

## Phase 1: Pipeline Setup (THIS PHASE)

### Goal
Establish Python → C++ → CUDA compilation and testing pipeline.

### Tasks
- [x] Verify CUDA Toolkit installation
- [ ] Install pybind11
- [ ] Install CMake (if not present)
- [ ] Create project structure
- [ ] Write "Hello World" CUDA kernel
- [ ] Create pybind11 binding
- [ ] Write Python test

### Test Criterion
```python
import planar_cuda
result = planar_cuda.add_vectors([1, 2, 3], [4, 5, 6])
assert result == [5, 7, 9]  # ✅ Pipeline works!
```

### Directory Structure
```
Hackathon-Nov-25-Heilbronn43/
├── src/
│   ├── LCNv1/              # Existing Python implementation
│   ├── cuda/               # NEW: CUDA kernels
│   │   ├── geometry.cuh
│   │   ├── spatial_grid.cu
│   │   └── solver_kernel.cu
│   ├── cpp_binding/        # NEW: pybind11 interface
│   │   └── binding.cpp
│   └── cuda_utils/         # NEW: Helper utilities
│       └── vector_ops.cu   # Phase 1: Hello World
├── tests/
│   └── cuda_tests/         # NEW: GPU tests
│       └── test_phase1_pipeline.py
├── CMakeLists.txt          # NEW: Build configuration
└── setup.py                # NEW: Python package setup
```

---

## Phase 2: Geometry Kernel Migration

### Goal
Port integer geometry functions to CUDA with 100% parity.

### Implementation
**File**: `src/cuda/geometry.cuh`
```cpp
__device__ int cross_product(int ax, int ay, int bx, int by);
__device__ bool segments_intersect(Point p1, Point p2, Point q1, Point q2);
```

**File**: `src/cuda/geometry_kernel.cu`
```cpp
__global__ void count_crossings_kernel(
    const Edge* edges, int num_edges,
    const Point* positions,
    int* crossing_counts
);
```

### Test Criterion
```python
# Load sol-15-nodes-5-planar.json
solver = planar_cuda.GeometrySolver(nodes, edges)
max_k = solver.count_all_crossings()
assert max_k <= 5  # Must match Python implementation
```

---

## Phase 3: GPU Memory Management

### Goal
Keep graph data on GPU, minimize PCIe transfers.

### Implementation
**File**: `src/cpp_binding/cuda_graph.cpp`
```cpp
class CudaGraph {
    thrust::device_vector<Point> d_positions;
    thrust::device_vector<Edge> d_edges;
    
public:
    CudaGraph(const std::vector<Point>& nodes, 
              const std::vector<Edge>& edges);
    void update_node_position(int node_id, int x, int y);
    int calculate_total_crossings();
};
```

### Test Criterion
```python
graph = planar_cuda.CudaGraph(nodes, edges)
graph.update_node_position(5, 100, 200)
energy = graph.calculate_total_crossings()
# Verify no memory leaks, correct results
```

---

## Phase 4: CUDA-Accelerated SA Solver

### Goal
Implement full SA loop in C++ with CUDA kernels for Δ_E.

### Implementation
**File**: `src/cpp_binding/sa_solver.cpp`
```cpp
class CudaSASolver {
public:
    void run_annealing(int steps, float initial_temp, float cooling_rate);
    std::vector<Point> get_final_layout();
    int get_current_k();
};
```

**File**: `src/cuda/solver_kernel.cu`
```cpp
__global__ void compute_delta_energy_kernel(
    int moved_node,
    Point new_pos,
    const Edge* incident_edges,
    const Point* positions,
    const SpatialHash* grid,
    int* delta_crossings
);
```

### Test Criterion
```python
solver = planar_cuda.CudaSASolver(nodes, edges)
solver.run_annealing(steps=10000, temp=100.0)
final_layout = solver.get_final_layout()
k = solver.get_current_k()
assert k < initial_k  # Must show improvement
```

---

## Performance Targets

| Benchmark | Python (Numba) | CUDA Target | Speedup |
|-----------|----------------|-------------|---------|
| 15 nodes  | 9,524 it/s     | 50,000+ it/s| 5-10x   |
| 70 nodes  | ~500 it/s      | 10,000+ it/s| 20x     |
| 625 nodes | N/A (too slow) | 1,000+ it/s | ∞       |

---

## Development Principles

1. **Red-Green-Refactor**: Write test first, make it pass, optimize
2. **Integer-Only Geometry**: No floating-point in CUDA kernels
3. **Minimize PCIe Transfers**: Upload once, compute on GPU, download once
4. **Incremental Validation**: Each phase must pass all previous tests

---

## Current Status

**Active Phase**: Phase 1 - Pipeline Setup  
**Blocked By**: CMakeLists.txt creation, pybind11 installation  
**Next Milestone**: Successfully compile and run `test_phase1_pipeline.py`

---

## Dependencies

### Required Tools
- CUDA Toolkit 12.6+ (✅ Installed)
- Visual Studio 2019+ with C++ tools
- CMake 3.18+
- Python 3.10+ (✅ Ready)

### Required Packages
```bash
pip install pybind11 pytest numpy
```

### Build Command (Future)
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
cmake --build . --config Release
```

---

## Success Metrics

- ✅ Phase 1: Python can call CUDA function
- ⬜ Phase 2: GPU geometry matches CPU geometry (100% parity)
- ⬜ Phase 3: Graph lives on GPU, no memory leaks
- ⬜ Phase 4: SA solver faster than Numba by 10x+

**When all phases complete**: Ready for 625-node challenge! 🚀
