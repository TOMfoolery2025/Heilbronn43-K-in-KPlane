這是一份針對「最小化局部交叉數 (LCN) 求解器」的完整軟體開發指南。本指南嚴格遵循 **OOA (分析) $\rightarrow$ OOD (設計) $\rightarrow$ OOP (實作)** 的流程，並以 **TDD (測試驅動開發)** 為執行核心。

---

# 專案代號：Project Planar-X 開發指南

## 核心哲學
1.  **數學即真理 (Math is Truth):** 幾何運算均使用**整數運算**，嚴禁浮點數比較。
2.  **測試優先 (Test First):** 沒有通過測試的程式碼不存在。
3.  **增量優化 (Incremental Optimization):** 先求正確，再求 $O(1)$ 加速。

---

## 第一階段：OOA 物件導向分析 (Domain Modeling)

在這個階段，我們識別問題領域中的實體 (Entities) 與邊界 (Boundaries)。

### 1. 領域模型 (Domain Model)
*   **Graph (圖):** 一組節點 (Nodes) 和邊 (Edges) 的拓撲結構。此結構在優化過程中不變。
*   **Embedding (佈局):** 圖在 2D 整數網格上的具體表現。這是我們改變與優化的對象。
*   **Metric (度量):** 衡量 Embedding 好壞的標準，即 $LCN$ (局部交叉數) 和 $Cost$ (總能量)。
*   **Move (移動):** 改變 Embedding 的原子操作 (如：將節點 A 移到座標 $(x, y)$)。

### 2. 業務規則 (Business Rules)
*   **Grid Constraint:** 所有座標必須是整數 $\mathbb{Z}^2$。
*   **Exclusion Constraint:** 任意兩點不得重疊 (Collision)。
*   **Edge Definition:** 線段由兩個整數點定義，交叉判定需處理「共線」與「端點接觸」的邊界情況。

---

## 第二階段：OOD 物件導向設計 (System Architecture)

我們採用 **分層架構 (Layered Architecture)** 配合 **策略模式 (Strategy Pattern)**。

### 1. 類別圖設計 (Class Blueprint)

*   **Layer 1: 基礎幾何 (Primitives)**
    *   `Point(x: int, y: int)`: Value Object，不可變 (Immutable)。
    *   `GeometryCore`: Static Utility，包含 `cross_product`, `segments_intersect`。

*   **Layer 2: 狀態管理 (State)**
    *   `GraphData`: 唯讀，儲存拓撲結構 (Adjacency List)。
    *   `GridState`: 儲存 `node_id -> Point` 的映射。負責執行「移動」並維護「邊界框」。
    *   `SpatialHash`: 加速結構。觀察 `GridState` 的變化，維護 `Cell -> List[EdgeID]` 的索引。

*   **Layer 3: 評估與策略 (Evaluation)**
    *   `ICostFunction` (Interface): 定義 `calculate(state)` 與 `calculate_delta(move)`。
    *   `SoftMaxCost`: 實作 PDF 中的 $\sum k^2$ 能量函數。

*   **Layer 4: 求解器 (Solver)**
    *   `AnnealingEngine`: 控制溫度 $(T)$ 與迭代次數。
    *   `MoveGenerator`: 根據溫度決定生成 `ShiftMove` 或 `SwapMove`。

---

## 第三階段：OOP + TDD 實作路徑 (Implementation Roadmap)

請依照以下四個 Sprint 進行開發。每個 Sprint 必須先寫測試 (Red)，再寫程式 (Green)。

### Sprint 1: 絕對真理 —— 幾何核心
**目標:** 確保交叉判定 100% 正確。利用 `sol-15-nodes-5-planar.json` 作為 Golden Test。

1.  **建立測試 `tests/test_geometry.py`**:
    *   `test_intersection_basic`: 十字交叉 (True)。
    *   `test_intersection_endpoint`: V字型連接 (False)。
    *   **整合測試:** 讀取 `sol-15-nodes-5-planar.json`，計算其所有邊的交叉數。若你的幾何邏輯正確，最大交叉數應 $\le 5$ (假設該檔名代表 LCN=5)。
2.  **實作 `src/geometry.py`**:
    *   實作 `cross_product` (叉積)。
    *   實作 `check_intersection(p1, p2, q1, q2)`。
    *   *注意:* 必須使用整數運算，避免 `float` 除法。

### Sprint 2: 空間加速 —— Spatial Hashing
**目標:** 將查詢鄰近邊的時間複雜度降至 $O(1)$。利用 `15-nodes.json` 驗證。

1.  **建立測試 `tests/test_spatial.py`**:
    *   `test_rasterization`: 輸入線段 $(0,0) \to (0, 10)$，Cell Size=5，應佔用 Grid $(0,0), (0,1), (0,2)$。
    *   **一致性測試:** 讀取 `15-nodes.json`。
        *   方法 A: 雙層迴圈暴力算交叉數。
        *   方法 B: 將邊插入 Spatial Hash 後查詢候選邊再算交叉數。
        *   Assert: 方法 A 結果 == 方法 B 結果。
2.  **實作 `src/spatial_index.py`**:
    *   Class `SpatialHash`: 使用 Dictionary `{(x,y): [edge_ids]}`。
    *   實作 Bresenham 演算法或 AABB 碰撞檢測來填充 Grid。

### Sprint 3: 能量與增量更新 —— The Heartbeat
**目標:** 實作高效的 $\Delta E$ 計算。這是 SA 演算法能否跑得快的關鍵。

1.  **建立測試 `tests/test_energy.py`**:
    *   `test_delta_update`:
        *   載入 `15-nodes.json`，計算全圖能量 $E_{total}$。
        *   選一個點 $v$，移動到新位置。
        *   計算 `delta = cost_calculator.get_delta(...)`。
        *   實際更新 `state`，重算全圖能量 $E'_{total}$。
        *   Assert $E_{total} + delta == E'_{total}$。
2.  **實作 `src/cost.py`**:
    *   Class `SoftMaxCost`: 實作公式 $W_{cross} \cdot \sum k^p + W_{len} \cdot \sum len^2$。
    *   重點優化: `get_delta` 只需查詢與移動點相連的邊 (Incident Edges)。

### Sprint 4: 模擬退火 —— The Brain
**目標:** 串接所有組件，開始尋找最佳解。

1.  **建立測試 `tests/test_solver.py`**:
    *   `test_improvement`: 輸入一個簡單的混亂圖 (例如 5 個點隨機亂放)，設定高迭代次數，Assert 最終能量低於初始能量。
2.  **實作 `src/solver.py`**:
    *   Class `SimulatedAnnealing`:
        *   冷卻排程 (Cooling Schedule): $T_{k+1} = \alpha T_k$。
        *   Metropolis 準則: $P(accept) = e^{-\Delta E / T}$。
    *   實作 `MoveStrategy`: 高溫時大幅度 `Shift`，低溫時局部 `Swap`。

---

## 專案結構建議 (Project Layout)

```text
planar-x/
├── data/
│   ├── 15-nodes.json              # 原始測試資料 (Input)
│   └── sol-15-nodes-5-planar.json # 驗證用解答 (Ground Truth)
├── src/
│   ├── __init__.py
│   ├── geometry.py        # 純函數幾何運算
│   ├── graph.py           # 資料模型 (Graph, GridState)
│   ├── spatial_index.py   # Spatial Hash 實作
│   ├── cost.py            # 能量函數與 Delta 計算
│   └── solver.py          # 模擬退火主迴圈
├── tests/
│   ├── test_geometry.py   # Sprint 1
│   ├── test_spatial.py    # Sprint 2
│   ├── test_energy.py     # Sprint 3
│   └── test_integration.py# Sprint 4
└── main.py                # 程式進入點 (Load -> Init -> Solve -> Save)
```

## 開發 Checkpoints (檢核點)

1.  **Checkpoint Alpha:** 執行 `pytest tests/test_geometry.py` 全過。確認你能正確判讀 JSON 檔案。
2.  **Checkpoint Beta:** 執行 `pytest tests/test_energy.py` 全過。這證明你的 $O(1)$ 更新邏輯是數學上正確的。
3.  **Checkpoint Release:** 執行 `main.py 15-nodes.json`，程式能在 10 秒內跑完，且輸出的 JSON 透過 `GeometryUtils` 驗證其 LCN 顯著降低。

這份指南不僅是開發順序，更是品質保證的流程。請優先處理 **tests/** 資料夾中的內容，因為 `15-nodes.json` 是你目前唯一可信的規格書。