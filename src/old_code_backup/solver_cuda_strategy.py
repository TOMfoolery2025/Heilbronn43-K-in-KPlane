"""
GPU 加速的求解器策略
結合 CUDA C++ 核心和 Python 邏輯
"""
import sys
import os
sys.path.insert(0, 'src')

# 修復 Windows CUDA DLL 加載問題
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
    ]
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path):
            try:
                os.add_dll_directory(cuda_path)
            except:
                pass

import numpy as np
import json
import random
import math

from solver_strategy import ISolverStrategy, SolverFactory
from graph import GraphData, GridState
from geometry import Point

try:
    from geometry_cuda import cuda_geometry
    HAS_CUDA = cuda_geometry.use_gpu
except ImportError:
    HAS_CUDA = False
    print("Warning: CUDA acceleration not available")


class CUDASolverStrategy(ISolverStrategy):
    """
    GPU 加速的求解器策略
    
    關鍵優化:
    1. CUDA kernels 用於批量交叉檢測
    2. GPU 並行計算能量
    3. CPU 負責模擬退火邏輯
    4. 數據在 GPU 上盡量保持，減少傳輸
    """
    
    def __init__(self, w_cross=100.0, w_len=1.0, power=2):
        """
        初始化 GPU 求解器
        
        Args:
            w_cross: 交叉懲罰權重
            w_len: 邊長懲罰權重
            power: 交叉懲罰指數
        """
        self.w_cross = w_cross
        self.w_len = w_len
        self.power = power
        
        self.graph = None
        self.state = None
        
        # GPU 狀態
        self.use_gpu = HAS_CUDA
        self.edges_array = None
        self.positions_array = None
        
        if not self.use_gpu:
            print("⚠️ GPU not available, falling back to CPU")
    
    def load_from_json(self, json_path):
        """從 JSON 加載圖"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 構建圖
        nodes = data['nodes']
        edges = data['edges']
        width = data.get('width', 1000000)
        height = data.get('height', 1000000)
        
        num_nodes = len(nodes)
        edge_list = []
        
        for edge in edges:
            src = edge['source']
            tgt = edge['target']
            edge_list.append((src, tgt))
        
        self.graph = GraphData(num_nodes, edge_list)
        
        # 初始位置
        positions = {node['id']: Point(node['x'], node['y']) for node in nodes}
        self.state = GridState(positions, width, height)
        
        # 準備 GPU 數組
        self._prepare_gpu_arrays()
    
    def _prepare_gpu_arrays(self):
        """準備 GPU 使用的 NumPy 數組"""
        # 邊列表: [num_edges, 2]
        self.edges_array = np.array(
            [(src, tgt) for src, tgt in self.graph.edges],
            dtype=np.int32
        )
        
        # 位置數組: [num_nodes, 2]
        self.positions_array = np.zeros((self.graph.num_nodes, 2), dtype=np.int32)
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            self.positions_array[node_id] = [pos.x, pos.y]
    
    def _update_positions_array(self):
        """更新位置數組"""
        for node_id in range(self.graph.num_nodes):
            pos = self.state.get_position(node_id)
            self.positions_array[node_id] = [pos.x, pos.y]
    
    def _calculate_energy_gpu(self):
        """
        使用 GPU 計算總能量
        
        Returns:
            (energy, k, total_crossings)
        """
        if not self.use_gpu:
            return self._calculate_energy_cpu()
        
        # 使用 CUDA kernel 計算交叉
        edge_crossings = cuda_geometry.count_all_crossings_gpu(
            self.edges_array,
            self.positions_array
        )
        
        # 計算交叉能量
        k = int(np.max(edge_crossings)) if len(edge_crossings) > 0 else 0
        total_crossings = int(np.sum(edge_crossings)) // 2
        crossing_energy = self.w_cross * np.sum(edge_crossings ** self.power)
        
        # 計算長度能量（CPU，很快）
        length_energy = 0.0
        for src, tgt in self.graph.edges:
            p1 = self.positions_array[src]
            p2 = self.positions_array[tgt]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length_energy += dx * dx + dy * dy
        
        length_energy *= self.w_len
        
        total_energy = crossing_energy + length_energy
        
        return (total_energy, k, total_crossings)
    
    def _calculate_energy_cpu(self):
        """CPU fallback"""
        from cost import SoftMaxCost
        
        cost_fn = SoftMaxCost(self.w_cross, self.w_len, self.power)
        energy = cost_fn.calculate(self.graph, self.state)
        k, total_crossings = cost_fn.get_crossing_stats(self.graph, self.state)
        
        return (energy, k, total_crossings)
    
    def _calculate_delta_gpu(self, node_id, new_pos):
        """
        計算移動節點後的能量變化（GPU 加速）
        
        這裡使用混合策略:
        - 小規模變化: GPU
        - 大規模變化: CPU（避免數據傳輸開銷）
        """
        # 獲取受影響的邊
        incident_edges = self.graph.get_incident_edges(node_id)
        
        # 如果受影響邊很少，CPU 可能更快（避免 GPU 傳輸開銷）
        if len(incident_edges) < 10:
            return self._calculate_delta_cpu(node_id, new_pos)
        
        # TODO: 實現 GPU delta 計算
        # 目前使用 CPU fallback
        return self._calculate_delta_cpu(node_id, new_pos)
    
    def _calculate_delta_cpu(self, node_id, new_pos):
        """CPU delta 計算"""
        from cost import SoftMaxCost
        
        cost_fn = SoftMaxCost(self.w_cross, self.w_len, self.power)
        delta = cost_fn.calculate_delta(self.graph, self.state, node_id, new_pos)
        
        return delta
    
    def solve(self, iterations=1000, initial_temp=50.0, 
              cooling_rate=0.995, reheat_threshold=500, **kwargs):
        """
        模擬退火優化（CPU 邏輯 + GPU 計算）
        
        Args:
            iterations: 迭代次數
            initial_temp: 初始溫度
            cooling_rate: 降溫率
            reheat_threshold: 重新加熱閾值
        
        Returns:
            優化結果字典
        """
        temp = initial_temp
        current_energy, current_k, current_crossings = self._calculate_energy_gpu()
        
        best_energy = current_energy
        best_state = self.state.copy()
        
        no_improvement_count = 0
        accepted = 0
        
        print(f"GPU Solver - Initial: E={current_energy:.0f}, K={current_k}, X={current_crossings}")
        
        for i in range(iterations):
            # 隨機選擇節點和新位置
            node_id = random.randint(0, self.graph.num_nodes - 1)
            old_pos = self.state.get_position(node_id)
            
            # 在附近移動
            radius = max(10, int(temp))
            new_x = old_pos.x + random.randint(-radius, radius)
            new_y = old_pos.y + random.randint(-radius, radius)
            new_pos = Point(new_x, new_y)
            
            # 計算 delta
            delta = self._calculate_delta_cpu(node_id, new_pos)
            
            # Metropolis 準則
            accept = False
            if delta < 0:
                accept = True
            elif temp > 0:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True
            
            if accept:
                # 接受移動
                self.state.set_position(node_id, new_pos)
                self.positions_array[node_id] = [new_pos.x, new_pos.y]
                current_energy += delta
                accepted += 1
                
                # 定期重新計算精確能量（避免累積誤差）
                if i % 100 == 0:
                    current_energy, current_k, current_crossings = self._calculate_energy_gpu()
                
                # 更新最佳
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_state = self.state.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # 降溫
            temp *= cooling_rate
            
            # 重新加熱
            if no_improvement_count >= reheat_threshold:
                temp = initial_temp * 0.5
                no_improvement_count = 0
            
            # 進度報告
            if (i + 1) % 100 == 0:
                current_energy, current_k, current_crossings = self._calculate_energy_gpu()
                print(f"  Iter {i+1}/{iterations}: E={current_energy:.0f}, K={current_k}, "
                      f"X={current_crossings}, T={temp:.2f}, Accept={accepted/(i+1)*100:.1f}%")
        
        # 恢復最佳狀態
        self.state = best_state
        self._update_positions_array()
        
        final_energy, final_k, final_crossings = self._calculate_energy_gpu()
        
        return {
            'energy': final_energy,
            'k': final_k,
            'total_crossings': final_crossings,
            'iterations': iterations,
            'acceptance_rate': accepted / iterations
        }
    
    def get_current_stats(self):
        """獲取當前統計"""
        energy, k, total_crossings = self._calculate_energy_gpu()
        
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
if HAS_CUDA:
    SolverFactory.register_strategy('cuda', CUDASolverStrategy)
    print("[OK] CUDA solver strategy registered")
else:
    print("⚠️ CUDA solver strategy not available (no GPU)")
