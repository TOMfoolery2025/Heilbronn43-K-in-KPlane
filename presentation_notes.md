# K-Planar Graph Minimizer - Presentation Notes

## 1. Solver Strategies
All strategies implement the **Simulated Annealing** meta-heuristic to find a global minimum for the graph's energy function. The energy function minimizes edge crossings while maintaining graph structure.

### **1. Legacy (Baseline)**
*   **Algorithm**: **Simulated Annealing** with additional forces:
    *   **Node Repulsion**: Pushes nodes apart to prevent overlap (Force-Directed Graph Drawing principle).
    *   **Spring Attraction**: Pulls connected nodes together to minimize edge length.
*   **Technology**: Pure Python.
*   **Description**: The original implementation. It calculates edge crossings and energy using standard Python loops.
*   **Pros**: Simple, easy to debug, serves as a correctness baseline.
*   **Cons**: Very slow for large graphs ($O(E^2)$ complexity in interpreted Python).

### **2. New (Vectorized)**
*   **Algorithm**: **Vectorized Simulated Annealing**.
    *   Instead of iterating edge-by-edge, it calculates the entire energy matrix in one go using Linear Algebra.
    *   Uses **Broadcasting** to compute all pairwise edge intersections simultaneously.
*   **Technology**: Python + NumPy.
*   **Description**: Replaces explicit loops with vectorized array operations. Calculates all interactions simultaneously using matrix math.
*   **Pros**: Significantly faster than Legacy due to C-level optimizations in NumPy.
*   **Cons**: Still bounded by single-core CPU performance.

### **3. Numba (JIT Compiled)**
*   **Algorithm**: **JIT-Compiled Simulated Annealing**.
    *   Uses the same logic as Legacy but compiles the hot loops (energy calculation) into optimized machine code at runtime.
    *   Bypasses the Python Interpreter overhead for the critical $O(E^2)$ loops.
*   **Technology**: Python + Numba (Just-In-Time Compiler).
*   **Description**: Compiles the Python calculation code into optimized machine code at runtime.
*   **Pros**: Near C++ performance on the CPU. Excellent for medium-sized graphs.
*   **Cons**: Compilation overhead on the first run; limited to CPU parallelism.

### **4. CUDA (GPU Accelerated)**
*   **Algorithm**: **Parallel Simulated Annealing**.
    *   **Massive Parallelism**: Offloads the $O(E^2)$ crossing counting task to the GPU.
    *   **Kernel Execution**: Each GPU thread calculates the intersections for a single edge against all other edges in parallel.
    *   **Metropolis Update**: The acceptance logic is handled efficiently after the heavy lifting is done on the GPU.
*   **Technology**: Python + CuPy + NVIDIA CUDA.
*   **Description**: Offloads the massively parallel task of counting edge crossings to the GPU.
*   **Pros**: **Fastest strategy for large graphs.** Can handle thousands of edges in milliseconds.
*   **Cons**: Requires an NVIDIA GPU; overhead of data transfer makes it slower for very small graphs.

---

## 2. Visualizer Layout & Indicators

### **Layout**
*   **Sidebar (Left)**:
    *   **Controls**: Load JSON data, select solver strategy, search for nodes.
    *   **Stats Panel**: Real-time display of the K-value (max crossings per edge) and Total Crossings.
*   **Main View (Right)**:
    *   **Interactive Plot**: Zoom, pan, and drag nodes manually.
    *   **Dynamic Scaling**: Automatically adjusts the view to fit the graph.

### **Edge Coloring (Visual Feedback)**
The visualizer uses color to highlight problem areas in the graph:

*   **<span style="color:gray">Gray Edges</span>**: **Low Conflict**.
    *   These edges have **3 or fewer crossings**. They are considered "acceptable" in the current layout.
    *   *Condition:* `crossings <= 3`

*   **<span style="color:red">Red Edges</span>**: **High Conflict**.
    *   These edges have **more than 3 crossings**.
    *   **Purpose**: They visually indicate where the solver is struggling or where the graph is most tangled. The optimizer focuses on reducing these red edges to minimize the global energy.
    *   *Condition:* `crossings > 3`
