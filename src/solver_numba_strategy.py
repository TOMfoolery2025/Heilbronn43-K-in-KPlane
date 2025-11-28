"""
Numba JIT 加速的求解器策略
使用 CPU JIT 編譯獲得接近 C 的性能
"""
import sys
sys.path.insert(0, 'src')

import numpy as np
import json
import random
import math
from numba import jit, prange
import time

from solver_strategy import ISolverStrategy, SolverFactory
from graph import GraphData, GridState
from geometry import Point


# Numba JIT 編譯的核心函數
@jit(nopython=True)
def cross_product_numba(ox, oy, ax, ay, bx, by):
    """JIT 編譯的叉積計算"""
    dx1 = ax - ox
    dy1 = ay - oy
    dx2 = bx - ox
    dy2 = by - oy
    return dx1 * dy2 - dy1 * dx2


@jit(nopython=True)
def segments_intersect_numba(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
    """JIT 編譯的線段相交檢測"""
    # 端點重疊不算相交
    if ((p1x == q1x and p1y == q1y) or (p1x == q2x and p1y == q2y) or
        (p2x == q1x and p2y == q1y) or (p2x == q2x and p2y == q2y)):
        return False
    
    cp1 = cross_product_numba(p1x, p1y, p2x, p2y, q1x, q1y)
    cp2 = cross_product_numba(p1x, p1y, p2x, p2y, q2x, q2y)
    cq1 = cross_product_numba(q1x, q1y, q2x, q2y, p1x, p1y)
    cq2 = cross_product_numba(q1x, q1y, q2x, q2y, p2x, p2y)
    
    return (cp1 * cp2 < 0) and (cq1 * cq2 < 0)


@jit(nopython=True, parallel=True)
def count_all_crossings_numba(edges, positions):
    """
    並行計算所有邊的交叉數
    
    Args:
        edges: (num_edges, 2) 邊的端點索引
        positions: (num_nodes, 2) 節點位置
    
    Returns:
        edge_crossings: (num_edges,) 每條邊的交叉數
    """
    num_edges = edges.shape[0]
    edge_crossings = np.zeros(num_edges, dtype=np.int32)
    
    # 並行處理每條邊
    for i in prange(num_edges):
        src_i = edges[i, 0]
        tgt_i = edges[i, 1]
        p1x = positions[src_i, 0]
        p1y = positions[src_i, 1]
        p2x = positions[tgt_i, 0]
        p2y = positions[tgt_i, 1]
        
        count = 0
        # 只檢查 j > i 避免重複
        for j in range(i + 1, num_edges):
            src_j = edges[j, 0]
            tgt_j = edges[j, 1]
            q1x = positions[src_j, 0]
            q1y = positions[src_j, 1]
            q2x = positions[tgt_j, 0]
            q2y = positions[tgt_j, 1]
            
            if segments_intersect_numba(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
                count += 1
        
        edge_crossings[i] = count
    
    # 第二次掃描補全（因為只計算了 j > i）
    for i in prange(num_edges):
        for j in range(i):
            if edge_crossings[j] > 0:  # j 與其他邊有交叉
                src_i = edges[i, 0]
                tgt_i = edges[i, 1]
                p1x = positions[src_i, 0]
                p1y = positions[src_i, 1]
                p2x = positions[tgt_i, 0]
                p2y = positions[tgt_i, 1]
                
                src_j = edges[j, 0]
                tgt_j = edges[j, 1]
                q1x = positions[src_j, 0]
                q1y = positions[src_j, 1]
                q2x = positions[tgt_j, 0]
                q2y = positions[tgt_j, 1]
                
                if segments_intersect_numba(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y):
                    edge_crossings[i] += 1
    
    return edge_crossings


@jit(nopython=True)
def calculate_energy_numba(edges, positions, edge_crossings, w_cross, w_len, power):
    """JIT 編譯的能量計算"""
    # 交叉能量
    crossing_energy = 0.0
    for k in edge_crossings:
        crossing_energy += k ** power
    crossing_energy *= w_cross
    
    # 長度能量
    length_energy = 0.0
    for i in range(edges.shape[0]):
        src = edges[i, 0]
        tgt = edges[i, 1]
        dx = positions[tgt, 0] - positions[src, 0]
        dy = positions[tgt, 1] - positions[src, 1]
        length_energy += dx * dx + dy * dy
    length_energy *= w_len
    
    return crossing_energy + length_energy


class NumbaSolverStrategy(ISolverStrategy):
    """
    Numba JIT 加速的求解器策略
    
    優勢:
    - 10-100x CPU 加速
    - 無需 GPU
    - 自動並行化
    - 編譯時優化
    """
    
    def __init__(self, w_cross=100.0, w_len=1.0, power=2):
        self.w_cross = w_cross
        self.w_len = w_len
        self.power = power
        
        self.graph = None
        self.state = None
        self.edges_array = None
        self.positions_array = None
        
        print("✅ Numba JIT 求解器初始化")
    
    def load_from_json(self, json_path):
        """從 JSON 加載圖"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        nodes = data['nodes']
        edges = data['edges']
        
        num_nodes = len(nodes)
        edge_list = []
        
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            edge_list.append((src, tgt))
        
        self.graph = GraphData(num_nodes, edge_list)
        
        positions = {node['id']: Point(node['x'], node['y']) for node in nodes}
        
        # 計算足夠大的邊界（允許移動）
        all_x = [node['x'] for node in nodes]
        all_y = [node['y'] for node in nodes]
        min_x = min(all_x) if all_x else 0
        max_x = max(all_x) if all_x else 0
        min_y = min(all_y) if all_y else 0
        max_y = max(all_y) if all_y else 0
        
        # 留出足夠的移動空間（+/- 10000）
        width = max_x - min_x + 20000
        height = max_y - min_y + 20000
        
        self.state = GridState(positions, width, height)
        
        self._prepare_arrays()
    
    def _prepare_arrays(self):
        """準備 NumPy 數組用於 Numba"""
        self.edges_array = np.array(
            [(src, tgt) for src, tgt in self.graph.edges],
            dtype=np.int32
        )
        
        self.positions_array = np.zeros((self.graph.num_nodes, 2), dtype=np.int32)
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            self.positions_array[node_id] = [pos.x, pos.y]
    
    def _update_positions_array(self):
        """同步位置數組"""
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            self.positions_array[node_id] = [pos.x, pos.y]
    
    def _calculate_energy_fast(self):
        """使用 Numba 加速計算能量"""
        edge_crossings = count_all_crossings_numba(self.edges_array, self.positions_array)
        
        k = int(np.max(edge_crossings)) if len(edge_crossings) > 0 else 0
        total_crossings = int(np.sum(edge_crossings)) // 2
        
        energy = calculate_energy_numba(
            self.edges_array,
            self.positions_array,
            edge_crossings,
            self.w_cross,
            self.w_len,
            self.power
        )
        
        return (energy, k, total_crossings)
    
    def solve(self, iterations=1000, initial_temp=50.0, 
              cooling_rate=0.995, reheat_threshold=500, **kwargs):
        """模擬退火優化"""
        print("\n🚀 Numba JIT 求解器開始...")
        print("預熱 JIT 編譯...")
        
        # 預熱：第一次調用會編譯，後續會很快
        _ = self._calculate_energy_fast()
        
        temp = initial_temp
        current_energy, current_k, current_crossings = self._calculate_energy_fast()
        
        best_energy = current_energy
        best_positions = self.positions_array.copy()
        
        no_improvement_count = 0
        accepted = 0
        
        print(f"初始狀態: E={current_energy:.0f}, K={current_k}, X={current_crossings}")
        
        start_time = time.time()
        
        for i in range(iterations):
            # 隨機移動
            node_id = random.randint(0, self.graph.num_nodes - 1)
            old_x, old_y = self.positions_array[node_id]
            
            radius = max(10, int(temp))
            new_x = old_x + random.randint(-radius, radius)
            new_y = old_y + random.randint(-radius, radius)
            
            # 嘗試移動
            self.positions_array[node_id] = [new_x, new_y]
            new_energy, _, _ = self._calculate_energy_fast()
            delta = new_energy - current_energy
            
            # Metropolis 準則
            accept = False
            if delta < 0:
                accept = True
            elif temp > 0:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True
            
            if accept:
                current_energy = new_energy
                # 直接更新內部狀態，跳過邊界檢查
                old_pos = self.state._positions.get(node_id)
                if old_pos and old_pos in self.state._location_to_node:
                    if self.state._location_to_node[old_pos] == node_id:
                        del self.state._location_to_node[old_pos]
                
                new_point = Point(int(new_x), int(new_y))
                self.state._positions[node_id] = new_point
                self.state._location_to_node[new_point] = node_id
                accepted += 1
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = self.positions_array.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # 恢復
                self.positions_array[node_id] = [old_x, old_y]
                no_improvement_count += 1
            
            # 降溫
            temp *= cooling_rate
            
            # 重新加熱
            if no_improvement_count >= reheat_threshold:
                temp = initial_temp * 0.5
                no_improvement_count = 0
            
            # 進度報告
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                current_energy, current_k, current_crossings = self._calculate_energy_fast()
                speed = (i + 1) / elapsed
                print(f"  [{i+1:>4}/{iterations}] E={current_energy:>8.0f}, K={current_k:>2}, "
                      f"X={current_crossings:>4}, T={temp:>5.2f}, "
                      f"Accept={accepted/(i+1)*100:>4.1f}%, Speed={speed:>6.1f} it/s")
        
        # 恢復最佳狀態
        self.positions_array = best_positions
        for node_id in range(self.graph.num_nodes):
            point = Point(int(self.positions_array[node_id, 0]),
                         int(self.positions_array[node_id, 1]))
            self.state._positions[node_id] = point
            self.state._location_to_node[point] = node_id
        
        final_energy, final_k, final_crossings = self._calculate_energy_fast()
        total_time = time.time() - start_time
        
        print(f"\n✅ 完成！總時間: {total_time:.2f}s, 速度: {iterations/total_time:.1f} it/s")
        
        return {
            'energy': final_energy,
            'k': final_k,
            'total_crossings': final_crossings,
            'iterations': iterations,
            'acceptance_rate': accepted / iterations,
            'time': total_time
        }
    
    def get_current_stats(self):
        """獲取當前統計"""
        self._update_positions_array()
        energy, k, total_crossings = self._calculate_energy_fast()
        
        return {
            'energy': energy,
            'k': k,
            'total_crossings': total_crossings
        }
    
    def export_to_json(self, output_path):
        """導出結果"""
        nodes = []
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            nodes.append({
                'id': node_id,
                'x': pos.x,
                'y': pos.y
            })
        
        edges = []
        for src, tgt in self.graph.edges:
            edges.append({
                'source': src,
                'target': tgt
            })
        
        result = {
            'nodes': nodes,
            'edges': edges
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)


# 註冊策略
SolverFactory.register_strategy('numba', NumbaSolverStrategy)
print("✅ Numba JIT 策略已註冊")
