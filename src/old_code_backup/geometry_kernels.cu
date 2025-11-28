/**
 * CUDA 核心：線段相交檢測加速
 * 
 * 編譯方式:
 * nvcc -c -o geometry_kernels.o geometry_kernels.cu -arch=sm_75 --ptx
 * 
 * 或者使用 PyCUDA/CuPy 動態編譯
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/**
 * 設備函數：計算整數叉積
 * 避免溢出使用 long long
 */
__device__ inline long long cross_product(
    int ox, int oy,
    int ax, int ay,
    int bx, int by
) {
    long long dx1 = (long long)(ax - ox);
    long long dy1 = (long long)(ay - oy);
    long long dx2 = (long long)(bx - ox);
    long long dy2 = (long long)(by - oy);
    return dx1 * dy2 - dy1 * dx2;
}

/**
 * 設備函數：檢查兩線段是否相交（嚴格）
 */
__device__ inline bool segments_intersect(
    int p1x, int p1y, int p2x, int p2y,
    int q1x, int q1y, int q2x, int q2y
) {
    // 檢查端點重疊（不算相交）
    if ((p1x == q1x && p1y == q1y) || (p1x == q2x && p1y == q2y) ||
        (p2x == q1x && p2y == q1y) || (p2x == q2x && p2y == q2y)) {
        return false;
    }
    
    // 計算四個叉積
    long long cp1 = cross_product(p1x, p1y, p2x, p2y, q1x, q1y);
    long long cp2 = cross_product(p1x, p1y, p2x, p2y, q2x, q2y);
    long long cq1 = cross_product(q1x, q1y, q2x, q2y, p1x, p1y);
    long long cq2 = cross_product(q1x, q1y, q2x, q2y, p2x, p2y);
    
    // 嚴格相交：兩組叉積異號
    return (cp1 * cp2 < 0) && (cq1 * cq2 < 0);
}

/**
 * Kernel 1: 批量線段相交檢測
 * 
 * 每個線程處理一對線段
 * 時間複雜度: O(1) 並行
 * 
 * @param p1_x, p1_y, p2_x, p2_y: 第一組線段端點
 * @param q1_x, q1_y, q2_x, q2_y: 第二組線段端點
 * @param results: 輸出結果（0 或 1）
 * @param n: 線段對數量
 */
extern "C" __global__
void segments_intersect_batch(
    const int* p1_x, const int* p1_y,
    const int* p2_x, const int* p2_y,
    const int* q1_x, const int* q1_y,
    const int* q2_x, const int* q2_y,
    int* results,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    results[idx] = segments_intersect(
        p1_x[idx], p1_y[idx], p2_x[idx], p2_y[idx],
        q1_x[idx], q1_y[idx], q2_x[idx], q2_y[idx]
    ) ? 1 : 0;
}

/**
 * Kernel 2: 計算每條邊的交叉數
 * 
 * 每個線程處理一條邊，檢查與所有其他邊的相交
 * 時間複雜度: O(E) 並行（每個線程 O(E)，但 E 個線程並行執行）
 * 
 * 注意：這是 naive 版本，適合中等規模圖
 * 大規模圖需要更複雜的分塊策略
 */
extern "C" __global__
void count_edge_crossings(
    const int* edges,           // [num_edges, 2] 邊的端點索引
    const int* positions_x,     // [num_nodes] 節點 x 坐標
    const int* positions_y,     // [num_nodes] 節點 y 坐標
    int* crossings,             // [num_edges] 輸出
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    
    // 獲取邊 i 的端點
    int src_i = edges[i * 2];
    int tgt_i = edges[i * 2 + 1];
    int p1x = positions_x[src_i];
    int p1y = positions_y[src_i];
    int p2x = positions_x[tgt_i];
    int p2y = positions_y[tgt_i];
    
    int count = 0;
    
    // 檢查與所有 j > i 的邊
    for (int j = i + 1; j < num_edges; j++) {
        int src_j = edges[j * 2];
        int tgt_j = edges[j * 2 + 1];
        int q1x = positions_x[src_j];
        int q1y = positions_y[src_j];
        int q2x = positions_x[tgt_j];
        int q2y = positions_y[tgt_j];
        
        if (segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y)) {
            count++;
        }
    }
    
    crossings[i] = count;
}

/**
 * Kernel 3: 優化版交叉計數（使用共享內存）
 * 
 * 將位置數據載入共享內存以減少全局內存訪問
 * 適合節點數 < 10000 的情況
 */
extern "C" __global__
void count_edge_crossings_optimized(
    const int* edges,
    const int* positions_x,
    const int* positions_y,
    int* crossings,
    int num_edges,
    int num_nodes
) {
    // 共享內存用於緩存位置（如果節點數不太大）
    extern __shared__ int shared_mem[];
    int* shared_x = shared_mem;
    int* shared_y = &shared_mem[num_nodes];
    
    // 協作加載位置到共享內存
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    for (int i = tid; i < num_nodes; i += total_threads) {
        shared_x[i] = positions_x[i];
        shared_y[i] = positions_y[i];
    }
    __syncthreads();
    
    // 處理邊
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    
    int src_i = edges[i * 2];
    int tgt_i = edges[i * 2 + 1];
    int p1x = shared_x[src_i];
    int p1y = shared_y[src_i];
    int p2x = shared_x[tgt_i];
    int p2y = shared_y[tgt_i];
    
    int count = 0;
    for (int j = i + 1; j < num_edges; j++) {
        int src_j = edges[j * 2];
        int tgt_j = edges[j * 2 + 1];
        int q1x = shared_x[src_j];
        int q1y = shared_y[src_j];
        int q2x = shared_x[tgt_j];
        int q2y = shared_y[tgt_j];
        
        if (segments_intersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y)) {
            count++;
        }
    }
    
    crossings[i] = count;
}

/**
 * Kernel 4: Delta 計算（移動一個節點後的增量）
 * 
 * 只計算受影響的邊
 * 時間複雜度: O(d * k) where d = degree, k = avg crossings
 */
extern "C" __global__
void calculate_delta_crossings(
    const int* edges,
    const int* positions_x,
    const int* positions_y,
    const int* incident_edges,  // 受影響的邊列表
    int num_incident,
    int node_id,
    int new_x,
    int new_y,
    int old_x,
    int old_y,
    int* old_crossings,         // 輸出：舊配置的交叉數
    int* new_crossings,         // 輸出：新配置的交叉數
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_incident) return;
    
    int edge_i = incident_edges[idx];
    int src_i = edges[edge_i * 2];
    int tgt_i = edges[edge_i * 2 + 1];
    
    // 舊位置
    int p1x_old = (src_i == node_id) ? old_x : positions_x[src_i];
    int p1y_old = (src_i == node_id) ? old_y : positions_y[src_i];
    int p2x_old = (tgt_i == node_id) ? old_x : positions_x[tgt_i];
    int p2y_old = (tgt_i == node_id) ? old_y : positions_y[tgt_i];
    
    // 新位置
    int p1x_new = (src_i == node_id) ? new_x : positions_x[src_i];
    int p1y_new = (src_i == node_id) ? new_y : positions_y[src_i];
    int p2x_new = (tgt_i == node_id) ? new_x : positions_x[tgt_i];
    int p2y_new = (tgt_i == node_id) ? new_y : positions_y[tgt_i];
    
    int count_old = 0;
    int count_new = 0;
    
    // 檢查與所有其他邊的相交
    for (int j = 0; j < num_edges; j++) {
        if (j == edge_i) continue;
        
        int src_j = edges[j * 2];
        int tgt_j = edges[j * 2 + 1];
        int q1x = positions_x[src_j];
        int q1y = positions_y[src_j];
        int q2x = positions_x[tgt_j];
        int q2y = positions_y[tgt_j];
        
        if (segments_intersect(p1x_old, p1y_old, p2x_old, p2y_old, q1x, q1y, q2x, q2y)) {
            count_old++;
        }
        
        if (segments_intersect(p1x_new, p1y_new, p2x_new, p2y_new, q1x, q1y, q2x, q2y)) {
            count_new++;
        }
    }
    
    old_crossings[idx] = count_old;
    new_crossings[idx] = count_new;
}

/**
 * Kernel 5: 空間哈希構建（並行）
 * 
 * 每個線程處理一條邊，將其插入到空間哈希表中
 * 使用原子操作處理衝突
 */
extern "C" __global__
void build_spatial_hash(
    const int* edges,
    const int* positions_x,
    const int* positions_y,
    int* hash_table,            // 哈希表 [hash_size]
    int* hash_counts,           // 每個桶的計數
    int cell_size,
    int hash_size,
    int num_edges
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_edges) return;
    
    int src = edges[edge_idx * 2];
    int tgt = edges[edge_idx * 2 + 1];
    
    int x1 = positions_x[src] / cell_size;
    int y1 = positions_y[src] / cell_size;
    int x2 = positions_x[tgt] / cell_size;
    int y2 = positions_y[tgt] / cell_size;
    
    // Bresenham 算法遍歷格子
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;
    
    int x = x1, y = y1;
    
    for (int step = 0; step < dx + dy + 1; step++) {
        // 計算哈希
        int hash = ((x * 73856093) ^ (y * 19349663)) % hash_size;
        if (hash < 0) hash += hash_size;
        
        // 使用原子操作插入
        int pos = atomicAdd(&hash_counts[hash], 1);
        // 這裡需要更複雜的結構來存儲實際的邊索引
        // 簡化版本僅計數
        
        // Bresenham 步進
        if (x == x2 && y == y2) break;
        
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

/**
 * 效能註記:
 * 
 * 1. segments_intersect_batch:
 *    - 理論加速: 100-1000x（完全並行）
 *    - 適用場景: 批量檢測大量線段對
 * 
 * 2. count_edge_crossings:
 *    - 理論加速: 10-100x（取決於 GPU 核心數）
 *    - 瓶頸: 每個線程仍需 O(E) 迭代
 * 
 * 3. count_edge_crossings_optimized:
 *    - 額外加速: 2-5x（共享內存減少全局訪問）
 *    - 限制: 節點數需 < 10000（共享內存大小）
 * 
 * 4. calculate_delta_crossings:
 *    - 關鍵優化: 只處理受影響的邊
 *    - 加速比: 50-500x（對於稀疏圖）
 * 
 * 5. build_spatial_hash:
 *    - 並行構建空間索引
 *    - 加速比: 10-50x
 */
