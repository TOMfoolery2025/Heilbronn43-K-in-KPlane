"""
CUDA 加速的幾何計算核心
使用 PyCUDA 或 CuPy 調用 .cu 文件中的 CUDA kernels
"""
import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("Warning: CuPy not installed. GPU acceleration disabled.")

from geometry import Point, GeometryCore


class CUDAGeometryCore:
    """
    GPU 加速的幾何計算核心
    
    關鍵優化:
    1. 批量線段相交檢測 - O(1) 在 GPU 上（並行）
    2. 空間哈希構建 - 並行化
    3. 交叉計數 - 並行歸約
    """
    
    # CUDA kernel 代碼（嵌入 Python，也可以單獨 .cu 文件）
    SEGMENTS_INTERSECT_KERNEL = r'''
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
        
        // 提取點坐標
        int p1x = p1_x[idx], p1y = p1_y[idx];
        int p2x = p2_x[idx], p2y = p2_y[idx];
        int q1x = q1_x[idx], q1y = q1_y[idx];
        int q2x = q2_x[idx], q2y = q2_y[idx];
        
        // 計算叉積 (整數運算)
        auto cross = [](int ox, int oy, int ax, int ay, int bx, int by) -> long long {
            long long dx1 = (long long)(ax - ox);
            long long dy1 = (long long)(ay - oy);
            long long dx2 = (long long)(bx - ox);
            long long dy2 = (long long)(by - oy);
            return dx1 * dy2 - dy1 * dx2;
        };
        
        // 檢查端點是否重疊
        if ((p1x == q1x && p1y == q1y) || (p1x == q2x && p1y == q2y) ||
            (p2x == q1x && p2y == q1y) || (p2x == q2x && p2y == q2y)) {
            results[idx] = 0;
            return;
        }
        
        // 計算四個叉積
        long long cp1 = cross(p1x, p1y, p2x, p2y, q1x, q1y);
        long long cp2 = cross(p1x, p1y, p2x, p2y, q2x, q2y);
        long long cq1 = cross(q1x, q1y, q2x, q2y, p1x, p1y);
        long long cq2 = cross(q1x, q1y, q2x, q2y, p2x, p2y);
        
        // 嚴格相交檢測（不包括端點接觸）
        bool intersects = (cp1 * cp2 < 0) && (cq1 * cq2 < 0);
        
        results[idx] = intersects ? 1 : 0;
    }
    '''
    
    # 交叉計數 kernel
    COUNT_CROSSINGS_KERNEL = r'''
    extern "C" __global__
    void count_edge_crossings(
        const int* edges,           // [num_edges, 2] 邊的端點索引
        const int* positions_x,     // [num_nodes] 節點 x 坐標
        const int* positions_y,     // [num_nodes] 節點 y 坐標
        int* crossings,             // [num_edges] 輸出每條邊的交叉數
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
        
        // 檢查與所有其他邊的相交
        for (int j = i + 1; j < num_edges; j++) {
            int src_j = edges[j * 2];
            int tgt_j = edges[j * 2 + 1];
            int q1x = positions_x[src_j];
            int q1y = positions_y[src_j];
            int q2x = positions_x[tgt_j];
            int q2y = positions_y[tgt_j];
            
            // 檢查端點重疊
            if ((p1x == q1x && p1y == q1y) || (p1x == q2x && p1y == q2y) ||
                (p2x == q1x && p2y == q1y) || (p2x == q2x && p2y == q2y)) {
                continue;
            }
            
            // 叉積計算
            auto cross = [](int ox, int oy, int ax, int ay, int bx, int by) -> long long {
                long long dx1 = (long long)(ax - ox);
                long long dy1 = (long long)(ay - oy);
                long long dx2 = (long long)(bx - ox);
                long long dy2 = (long long)(by - oy);
                return dx1 * dy2 - dy1 * dx2;
            };
            
            long long cp1 = cross(p1x, p1y, p2x, p2y, q1x, q1y);
            long long cp2 = cross(p1x, p1y, p2x, p2y, q2x, q2y);
            long long cq1 = cross(q1x, q1y, q2x, q2y, p1x, p1y);
            long long cq2 = cross(q1x, q1y, q2x, q2y, p2x, p2y);
            
            if ((cp1 * cp2 < 0) && (cq1 * cq2 < 0)) {
                count++;
            }
        }
        
        crossings[i] = count;
    }
    '''
    
    def __init__(self):
        """初始化 CUDA 核心"""
        self.use_gpu = HAS_CUPY
        
        if self.use_gpu:
            try:
                # 編譯 CUDA kernels
                self._compile_kernels()
                print("✅ CUDA kernels compiled successfully")
            except Exception as e:
                print(f"⚠️ CUDA compilation failed: {e}")
                self.use_gpu = False
    
    def _compile_kernels(self):
        """編譯 CUDA kernels"""
        if not HAS_CUPY:
            return
        
        # 使用 CuPy 的 RawKernel
        self.intersect_kernel = cp.RawKernel(
            self.SEGMENTS_INTERSECT_KERNEL,
            'segments_intersect_batch'
        )
        
        self.count_kernel = cp.RawKernel(
            self.COUNT_CROSSINGS_KERNEL,
            'count_edge_crossings'
        )
    
    def segments_intersect_batch_gpu(self, segments_p, segments_q):
        """
        批量檢測線段相交（GPU 版本）
        
        Args:
            segments_p: [(p1, p2), ...] 第一組線段
            segments_q: [(q1, q2), ...] 第二組線段（相同長度）
        
        Returns:
            np.array of bool: 相交結果
        """
        if not self.use_gpu:
            # Fallback to CPU
            return self._segments_intersect_batch_cpu(segments_p, segments_q)
        
        n = len(segments_p)
        
        # 準備數據
        p1_x = cp.array([p1.x for p1, _ in segments_p], dtype=cp.int32)
        p1_y = cp.array([p1.y for p1, _ in segments_p], dtype=cp.int32)
        p2_x = cp.array([p2.x for _, p2 in segments_p], dtype=cp.int32)
        p2_y = cp.array([p2.y for _, p2 in segments_p], dtype=cp.int32)
        
        q1_x = cp.array([q1.x for q1, _ in segments_q], dtype=cp.int32)
        q1_y = cp.array([q1.y for q1, _ in segments_q], dtype=cp.int32)
        q2_x = cp.array([q2.x for _, q2 in segments_q], dtype=cp.int32)
        q2_y = cp.array([q2.y for _, q2 in segments_q], dtype=cp.int32)
        
        results = cp.zeros(n, dtype=cp.int32)
        
        # 配置 GPU 執行
        block_size = 256
        grid_size = (n + block_size - 1) // block_size
        
        # 執行 kernel
        self.intersect_kernel(
            (grid_size,), (block_size,),
            (p1_x, p1_y, p2_x, p2_y, q1_x, q1_y, q2_x, q2_y, results, n)
        )
        
        # 傳回 CPU
        return cp.asnumpy(results).astype(bool)
    
    def count_all_crossings_gpu(self, edges, positions):
        """
        計算所有邊的交叉數（GPU 版本）
        
        Args:
            edges: np.array shape (num_edges, 2) 邊的端點索引
            positions: np.array shape (num_nodes, 2) 節點位置 (x, y)
        
        Returns:
            np.array: 每條邊的交叉數
        """
        if not self.use_gpu:
            return self._count_all_crossings_cpu(edges, positions)
        
        num_edges = len(edges)
        
        # 傳輸到 GPU
        edges_gpu = cp.array(edges, dtype=cp.int32)
        pos_x_gpu = cp.array(positions[:, 0], dtype=cp.int32)
        pos_y_gpu = cp.array(positions[:, 1], dtype=cp.int32)
        crossings_gpu = cp.zeros(num_edges, dtype=cp.int32)
        
        # 配置執行
        block_size = 256
        grid_size = (num_edges + block_size - 1) // block_size
        
        # 執行
        self.count_kernel(
            (grid_size,), (block_size,),
            (edges_gpu, pos_x_gpu, pos_y_gpu, crossings_gpu, num_edges)
        )
        
        # 傳回 CPU
        return cp.asnumpy(crossings_gpu)
    
    def _segments_intersect_batch_cpu(self, segments_p, segments_q):
        """CPU fallback"""
        results = []
        for (p1, p2), (q1, q2) in zip(segments_p, segments_q):
            results.append(GeometryCore.segments_intersect(p1, p2, q1, q2))
        return np.array(results, dtype=bool)
    
    def _count_all_crossings_cpu(self, edges, positions):
        """CPU fallback"""
        num_edges = len(edges)
        crossings = np.zeros(num_edges, dtype=np.int32)
        
        for i in range(num_edges):
            src_i, tgt_i = edges[i]
            p1 = Point(positions[src_i, 0], positions[src_i, 1])
            p2 = Point(positions[tgt_i, 0], positions[tgt_i, 1])
            
            for j in range(i + 1, num_edges):
                src_j, tgt_j = edges[j]
                q1 = Point(positions[src_j, 0], positions[src_j, 1])
                q2 = Point(positions[tgt_j, 0], positions[tgt_j, 1])
                
                if GeometryCore.segments_intersect(p1, p2, q1, q2):
                    crossings[i] += 1
                    crossings[j] += 1
        
        return crossings


# 全局實例
cuda_geometry = CUDAGeometryCore()
