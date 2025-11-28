# Hybrid Python + CUDA Development Guide (English Translation)

This is an excellent decision.

In Graph Drawing or physical simulation, Python is very suitable as a **Glue Language** to handle data input, parameter configuration, and visualization, but its interpreted nature becomes a huge bottleneck in tight loops with $O(N^2)$ or $O(N \log N)$ complexity.

Offloading the computational core to **CUDA (.cu)** and exposing it to Python through a C++ wrapper layer is the standard architecture for High-Performance Computing (HPC) (similar to the underlying implementation of PyTorch or TensorFlow).

Below is the development planning guide for adopting the **Hybrid Python + CUDA** architecture:

---

## 1. System Architecture: Python Frontend + CUDA Backend

We divide the system into three layers:

1.  **Python Layer (Frontend/Control Layer):**
    *   Responsible for reading `.json` files.
    *   Responsible for unit testing (pytest).
    *   Responsible for calling C++ modules and receiving final coordinates.
    *   Tools: `Python 3.10+`, `pytest`.

2.  **C++ Binding Layer (Interface Layer):**
    *   Acts as a bridge between Python and GPU.
    *   Manages GPU memory (allocation/free).
    *   Tools: **`pybind11`** (recommended) or `NanoBind`. This allows your C++ classes to be directly imported in Python.

3.  **CUDA Layer (Compute Core Layer):**
    *   Implements geometry core (`__device__` functions).
    *   Implements Spatial Hashing and parallel intersection detection (`__global__` kernels).
    *   Tools: `NVCC`, `Thrust` (CUDA's STL alternative), `CUDA C++`.

---

## 2. Core Object Responsibility Transfer (OOA Re-evaluation)

In the pure Python version, objects hold data; in the hybrid architecture, Python objects only hold pointers (handles) to C++ objects.

| Component | Python Responsibility | C++ / CUDA Responsibility |
| :--- | :--- | :--- |
| **Graph** | Parse JSON, pass node and edge lists to C++ | `struct GraphData` (Device Memory). Use `thrust::device_vector` to store coordinates and edges. |
| **Geometry** | **Removed** (only kept for test validation) | `__device__` functions: `cross_product`, `check_intersection`. |
| **Grid** | **Removed** | `Kernel Launch`: Parallel computation of which edges fall into which Grid Cell. |
| **Energy** | Receive current Energy value (for plotting) | `Reduction Kernel`: Parallel sum of all edge crossing penalties. |
| **Solver** | Call `solver.step(1000)` or `solver.solve()` | Implement SA's core loop (for performance, SA's Loop is best run in C++). |

---

## 3. Development Environment Setup

You need a `setup.py` or `CMakeLists.txt` to compile `.cu` files and generate `.so` (Linux) or `.pyd` (Windows) for Python to import.

**Recommended Structure:**
```text
project/
├── src/
│   ├── cuda/
│   │   ├── geometry.cuh       # __device__ helper functions
│   │   ├── spatial_grid.cu    # Grid kernels
│   │   └── solver_kernel.cu   # SA implementation
│   ├── cpp_binding/
│   │   └── binding.cpp        # pybind11 code
│   └── python/
│       └── visualizer.py
├── tests/
│   └── test_core.py           # Python calls CUDA for testing
├── CMakeLists.txt             # Compilation script
└── main.py
```

---

## 4. Development Steps (Step-by-Step with TDD)

### Phase 1: Establish Pybind11 + CUDA Skeleton
**Goal:** Enable Python to call a CUDA-written "Hello World" (e.g., vector addition).

1.  **Write C++ (`example.cu`)**: Write a simple `add_arrays` kernel.
2.  **Write Binding (`binding.cpp`)**: Use `PYBIND11_MODULE` to expose the function.
3.  **Write Python Test**:
    ```python
    import my_cuda_module
    def test_cuda_add():
        res = my_cuda_module.add([1, 2], [3, 4])
        assert res == [4, 6]
    ```
4.  **Compile and Test**: Ensure the pipeline works smoothly.

### Phase 2: Geometry Core Migration (Geometry Kernel)
**Goal:** Verify GPU calculations match CPU calculations.

1.  **Define C++ Struct**:
    ```cpp
    struct Point { int x, y; };
    struct Edge { int u, v; };
    ```
2.  **Write CUDA Device Function**:
    *   Port `cross_product` and `segments_intersect` to `geometry.cuh`, add `__device__` modifier.
3.  **Write Kernel**:
    *   `count_crossings_kernel`: Each Thread checks a pair of edges, calculates if they intersect.
4.  **Python TDD**:
    *   Read `sol-15-nodes...json`.
    *   Send coordinates to C++ module.
    *   Call `module.count_all_crossings()`.
    *   **Assert:** Result must be <= 5 (consistent with Python prototype result).

### Phase 3: Data Structure Transfer (Data on Device)
**Goal:** Avoid transferring List from Python to GPU every calculation (PCIe is too slow).

1.  **C++ Class `CudaGraph`**:
    *   Use `cudaMalloc` or `thrust::device_vector` to allocate memory in the Constructor.
    *   Provide `update_node_pos(node_id, x, y)` method.
2.  **Binding**:
    *   Expose `CudaGraph` class to Python.
    *   During Python initialization: `graph = CudaGraph(nodes_list, edges_list)` (data transferred only once).

### Phase 4: Simulated Annealing Core (The SA Loop)
**Key Decision:** For ultimate performance, **do not** write `for i in range(10000)` in Python and then call C++ each time. Python calling C++ has a small overhead that accumulates significantly.

**Correct Approach:** Write the entire SA Loop in C++, Python only calls `solve(steps=10000, temp=100)`.

1.  **C++ Implement SA**:
    *   Implement cooling loop on C++ side.
    *   Each iteration:
        1.  CPU (C++ Host) decides Move (Shift/Swap).
        2.  Call CUDA Kernel to calculate $\Delta E$ (only calculate affected edges).
        3.  Decide whether to accept based on Metropolis criterion.
        4.  If accepted, update coordinates in Device Memory.
2.  **Python TDD**:
    *   `graph.solve(steps=50000)`.
    *   `pos = graph.get_positions()`.
    *   Verify final `pos` crossing count.

---

## 5. CUDA-Specific Implementation Example

This is a simple `binding.cpp` conceptual example to give you an idea:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpu_solver.h" // Your CUDA header

namespace py = pybind11;

// This is the class you'll see in Python
class PySolver {
    GPUSolver* solver; // Pointer to actual CUDA instance

public:
    PySolver(std::vector<std::pair<int, int>> nodes, 
             std::vector<std::pair<int, int>> edges) {
        // Data from Python list converted to C++ vector, then passed to CUDA memory
        solver = new GPUSolver(nodes, edges);
    }

    ~PySolver() { delete solver; }

    void run_annealing(int steps, float initial_temp) {
        // This function will run for a long time, during which Python will be blocked
        solver->run(steps, initial_temp);
    }

    std::vector<std::pair<int, int>> get_layout() {
        return solver->download_results();
    }
    
    int get_current_energy() {
        return solver->calculate_full_energy();
    }
};

PYBIND11_MODULE(planar_cuda, m) {
    py::class_<PySolver>(m, "PlanarSolver")
        .def(py::init<std::vector<std::pair<int, int>>, std::vector<std::pair<int, int>>>())
        .def("run_annealing", &PySolver::run_annealing)
        .def("get_layout", &PySolver::get_layout)
        .def("get_energy", &PySolver::get_current_energy);
}
```

---

## 6. Benefits of This Development Approach

1.  **Test and Algorithm Separation:** You can write very complex test cases in Python without the pain of writing Unit Tests in C++.
2.  **Visualization:** During development, C++ can return coordinates every 1000 iterations, Python uses Matplotlib to draw in real-time, watching how the graph "untangles".
3.  **Performance:** $\Delta E$ calculations can be parallelized on GPU (checking one edge vs other candidate edges), which is hundreds of times faster than CPU.

---

## Summary of Your Action Items

1.  Install **CUDA Toolkit** ✅ (Already installed)
2.  Install **pybind11** (`pip install pybind11`)
3.  Don't worry about algorithms first, **establish the Python → C++ → CUDA compilation pipeline first**. This is the biggest hurdle; once crossed, the rest is just porting logic.

---

## 🚀 Quick Start

```powershell
# Run the automated build script
.\build_and_test.ps1
```

This will:
- ✅ Check prerequisites
- ✅ Install dependencies
- ✅ Build CUDA module
- ✅ Run all tests
- ✅ Verify Phase 1 complete

**Next**: Once Phase 1 passes, proceed to Phase 2 (Geometry Kernels)!

---

## 📚 Documentation Reference

- `QUICKSTART.md` - One-command build guide
- `BUILD_INSTRUCTIONS.md` - Detailed build instructions
- `ARCHITECTURE_SUMMARY.md` - Complete architecture overview
- `CUDA_HYBRID_ROADMAP.md` - Full development roadmap

Good luck with your CUDA acceleration journey! 🚀
