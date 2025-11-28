This is a complete software development guide for the "Local Crossing Number (LCN) Minimization Solver". This guide strictly follows the **OOA (Analysis) $\rightarrow$ OOD (Design) $\rightarrow$ OOP (Implementation)** process, with **TDD (Test-Driven Development)** as the execution core.

---

# Project Codename: Project Planar-X Development Guide

## Core Philosophy
1.  **Math is Truth:** All geometric operations use **integer arithmetic**, floating-point comparisons are strictly forbidden.
2.  **Test First:** Code without passing tests does not exist.
3.  **Incremental Optimization:** Correctness first, then $O(1)$ acceleration.

---

## Phase 1: OOA Object-Oriented Analysis (Domain Modeling)

In this phase, we identify entities and boundaries in the problem domain.

### 1. Domain Model
*   **Graph:** A topological structure consisting of nodes and edges. This structure remains unchanged during optimization.
*   **Embedding:** The concrete representation of the graph on a 2D integer grid. This is what we modify and optimize.
*   **Metric:** The standard for measuring embedding quality, namely $LCN$ (Local Crossing Number) and $Cost$ (total energy).
*   **Move:** An atomic operation that changes the embedding (e.g., moving node A to coordinates $(x, y)$).

### 2. Business Rules
*   **Grid Constraint:** All coordinates must be integers $\mathbb{Z}^2$.
*   **Exclusion Constraint:** No two points may overlap (collision).
*   **Edge Definition:** Line segments are defined by two integer points; intersection detection must handle "collinear" and "endpoint contact" boundary cases.

---

## Phase 2: OOD Object-Oriented Design (System Architecture)

We adopt a **Layered Architecture** combined with the **Strategy Pattern**.

### 1. Class Blueprint

*   **Layer 1: Primitives (Geometry)**
    *   `Point(x: int, y: int)`: Value Object, immutable.
    *   `GeometryCore`: Static Utility, contains `cross_product`, `segments_intersect`.

*   **Layer 2: State Management**
    *   `GraphData`: Read-only, stores topological structure (Adjacency List).
    *   `GridState`: Stores `node_id -> Point` mapping. Responsible for executing "moves" and maintaining "bounding box".
    *   `SpatialHash`: Acceleration structure. Observes `GridState` changes, maintains `Cell -> List[EdgeID]` index.

*   **Layer 3: Evaluation & Strategy**
    *   `ICostFunction` (Interface): Defines `calculate(state)` and `calculate_delta(move)`.
    *   `SoftMaxCost`: Implements the $\sum k^2$ energy function from the PDF.

*   **Layer 4: Solver**
    *   `AnnealingEngine`: Controls temperature $(T)$ and iteration count.
    *   `MoveGenerator`: Decides to generate `ShiftMove` or `SwapMove` based on temperature.

---

## Phase 3: OOP + TDD Implementation Roadmap

Follow these four Sprints for development. Each Sprint must write tests first (Red), then write code (Green).

### Sprint 1: Absolute Truth — Geometry Core
**Goal:** Ensure 100% correct intersection detection. Use `sol-15-nodes-5-planar.json` as the Golden Test.

1.  **Create test `tests/test_geometry.py`**:
    *   `test_intersection_basic`: Cross intersection (True).
    *   `test_intersection_endpoint`: V-shaped connection (False).
    *   **Integration test:** Read `sol-15-nodes-5-planar.json`, calculate crossing number for all edges. If your geometry logic is correct, the maximum crossing number should be $\le 5$ (assuming the filename indicates LCN=5).
2.  **Implement `src/geometry.py`**:
    *   Implement `cross_product` (cross product).
    *   Implement `check_intersection(p1, p2, q1, q2)`.
    *   *Note:* Must use integer arithmetic, avoid `float` division.

### Sprint 2: Spatial Acceleration — Spatial Hashing
**Goal:** Reduce neighbor edge query time complexity to $O(1)$. Validate using `15-nodes.json`.

1.  **Create test `tests/test_spatial.py`**:
    *   `test_rasterization`: Input segment $(0,0) \to (0, 10)$, Cell Size=5, should occupy Grid $(0,0), (0,1), (0,2)$.
    *   **Consistency test:** Read `15-nodes.json`.
        *   Method A: Brute force double loop to calculate crossing number.
        *   Method B: Insert edges into Spatial Hash, query candidate edges, then calculate crossing number.
        *   Assert: Method A result == Method B result.
2.  **Implement `src/spatial_index.py`**:
    *   Class `SpatialHash`: Use Dictionary `{(x,y): [edge_ids]}`.
    *   Implement Bresenham algorithm or AABB collision detection to populate Grid.

### Sprint 3: Energy & Incremental Updates — The Heartbeat
**Goal:** Implement efficient $\Delta E$ calculation. This is the key to whether the SA algorithm can run fast.

1.  **Create test `tests/test_energy.py`**:
    *   `test_delta_update`:
        *   Load `15-nodes.json`, calculate total graph energy $E_{total}$.
        *   Select a point $v$, move to new position.
        *   Calculate `delta = cost_calculator.get_delta(...)`.
        *   Actually update `state`, recalculate total graph energy $E'_{total}$.
        *   Assert $E_{total} + delta == E'_{total}$.
2.  **Implement `src/cost.py`**:
    *   Class `SoftMaxCost`: Implement formula $W_{cross} \cdot \sum k^p + W_{len} \cdot \sum len^2$.
    *   Key optimization: `get_delta` only needs to query edges connected to the moving point (Incident Edges).

### Sprint 4: Simulated Annealing — The Brain
**Goal:** Connect all components and start finding the optimal solution.

1.  **Create test `tests/test_solver.py`**:
    *   `test_improvement`: Input a simple chaotic graph (e.g., 5 randomly placed points), set high iteration count, Assert final energy is lower than initial energy.
2.  **Implement `src/solver.py`**:
    *   Class `SimulatedAnnealing`:
        *   Cooling Schedule: $T_{k+1} = \alpha T_k$.
        *   Metropolis Criterion: $P(accept) = e^{-\Delta E / T}$.
    *   Implement `MoveStrategy`: Large-scale `Shift` at high temperature, local `Swap` at low temperature.

---

## Project Structure Recommendation (Project Layout)

```text
planar-x/
├── data/
│   ├── 15-nodes.json              # Original test data (Input)
│   └── sol-15-nodes-5-planar.json # Validation solution (Ground Truth)
├── src/
│   ├── __init__.py
│   ├── geometry.py        # Pure function geometric operations
│   ├── graph.py           # Data model (Graph, GridState)
│   ├── spatial_index.py   # Spatial Hash implementation
│   ├── cost.py            # Energy function and Delta calculation
│   └── solver.py          # Simulated annealing main loop
├── tests/
│   ├── test_geometry.py   # Sprint 1
│   ├── test_spatial.py    # Sprint 2
│   ├── test_energy.py     # Sprint 3
│   └── test_integration.py# Sprint 4
└── main.py                # Program entry point (Load -> Init -> Solve -> Save)
```

## Development Checkpoints

1.  **Checkpoint Alpha:** Execute `pytest tests/test_geometry.py` all pass. Confirm you can correctly parse JSON files.
2.  **Checkpoint Beta:** Execute `pytest tests/test_energy.py` all pass. This proves your $O(1)$ update logic is mathematically correct.
3.  **Checkpoint Release:** Execute `main.py 15-nodes.json`, program completes within 10 seconds, and the output JSON verified through `GeometryUtils` shows significantly reduced LCN.

This guide is not only a development sequence but also a quality assurance process. Please prioritize content in the **tests/** folder, as `15-nodes.json` is currently your only trusted specification.