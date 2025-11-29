"""
CUDA Solver Strategy using CuPy
Implements parallel simulated annealing on GPU.
"""
import os
import sys

# Robustly setup CUDA paths for Windows BEFORE importing CuPy
_cuda_dll_handle = None

def _setup_cuda_paths():
    global _cuda_dll_handle
    print("[DEBUG] Searching for CUDA installation...")
    
    # Common default path
    default_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
    
    # Try to find if not in default
    if not os.path.exists(default_path):
        # Search in Program Files
        base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.exists(base):
            versions = os.listdir(base)
            if versions:
                # Pick the last one (likely newest)
                default_path = os.path.join(base, versions[-1], "bin")
                print(f"[DEBUG] Found CUDA version: {versions[-1]}")

    if os.path.exists(default_path):
        print(f"[DEBUG] CUDA bin path found: {default_path}")
        
        # Add to PATH
        if default_path not in os.environ["PATH"]:
            os.environ["PATH"] += ";" + default_path
            print("[DEBUG] Added CUDA bin to PATH")
        
        # Add to DLL Directory (Python 3.8+)
        if hasattr(os, 'add_dll_directory'):
            try:
                _cuda_dll_handle = os.add_dll_directory(default_path)
                print("[DEBUG] Added CUDA bin to DLL Directory (Handle kept alive)")
            except Exception as e:
                print(f"[DEBUG] Failed to add DLL directory: {e}")
        
        # Set CUDA_PATH if missing
        if "CUDA_PATH" not in os.environ:
            os.environ["CUDA_PATH"] = os.path.dirname(default_path)
            print(f"[DEBUG] Set CUDA_PATH to {os.environ['CUDA_PATH']}")
    else:
        print("[DEBUG] CUDA bin path NOT found!")

_setup_cuda_paths()

import cupy as cp
import numpy as np
import json
import random
import math
import time
from typing import Dict, Any, List

from LCNv1.strategies.base import ISolverStrategy
from LCNv1.core.graph import GraphData, GridState
from LCNv1.core.geometry import Point

# CUDA Kernel for counting crossings
# Each thread handles one edge and checks it against all other edges.
CROSSING_KERNEL_CODE = r'''
extern "C" __global__
void count_crossings_kernel(
    const int* edges,       // (E, 2)
    const float* positions, // (N, 2)
    int* out_counts,        // (E)
    int num_edges
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_edges) return;

    int src_i = edges[i * 2];
    int tgt_i = edges[i * 2 + 1];
    
    float p1x = positions[src_i * 2];
    float p1y = positions[src_i * 2 + 1];
    float p2x = positions[tgt_i * 2];
    float p2y = positions[tgt_i * 2 + 1];

    int count = 0;

    for (int j = 0; j < num_edges; j++) {
        if (i == j) continue;
        
        int src_j = edges[j * 2];
        int tgt_j = edges[j * 2 + 1];
        
        // Skip edges sharing a node (adjacent edges don't cross)
        if (src_i == src_j || src_i == tgt_j || tgt_i == src_j || tgt_i == tgt_j) continue;

        float q1x = positions[src_j * 2];
        float q1y = positions[src_j * 2 + 1];
        float q2x = positions[tgt_j * 2];
        float q2y = positions[tgt_j * 2 + 1];

        // Intersection Test
        // 1. Bounding box check (optimization)
        float min_p_x = min(p1x, p2x);
        float max_p_x = max(p1x, p2x);
        float min_q_x = min(q1x, q2x);
        float max_q_x = max(q1x, q2x);
        
        if (max_p_x < min_q_x || max_q_x < min_p_x) continue;
        
        float min_p_y = min(p1y, p2y);
        float max_p_y = max(p1y, p2y);
        float min_q_y = min(q1y, q2y);
        float max_q_y = max(q1y, q2y);
        
        if (max_p_y < min_q_y || max_q_y < min_p_y) continue;

        // 2. Cross Product Test
        float d1x = p2x - p1x;
        float d1y = p2y - p1y;
        float d2x = q2x - q1x;
        float d2y = q2y - q1y;

        float cp1 = (q1x - p1x) * d1y - (q1y - p1y) * d1x;
        float cp2 = (q2x - p1x) * d1y - (q2y - p1y) * d1x;
        
        if ((cp1 > 0 && cp2 > 0) || (cp1 < 0 && cp2 < 0)) continue; // Same side
        
        float cp3 = (p1x - q1x) * d2y - (p1y - q1y) * d2x;
        float cp4 = (p2x - q1x) * d2y - (p2y - q1y) * d2x;
        
        if ((cp3 > 0 && cp4 > 0) || (cp3 < 0 && cp4 < 0)) continue; // Same side
        
        count++;
    }
    
    out_counts[i] = count;
}
'''

class CUDASolverStrategy(ISolverStrategy):

    def __init__(self, w_cross=100.0, w_len=1.0, power=2):
        self.w_cross = w_cross
        self.w_len = w_len
        self.power = power
        
        self.graph = None
        self.state = None
        
        # GPU Arrays
        self.edges_gpu = None
        self.positions_gpu = None
        self.edge_crossings_gpu = None
        
        # Compile Kernel
        self.crossing_kernel = cp.RawKernel(CROSSING_KERNEL_CODE, 'count_crossings_kernel')
        
        print("[OK] CUDA solver initialized")

    def load_from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        nodes = data['nodes']
        edges = data['edges']
        
        num_nodes = len(nodes)
        edge_list = []
        for edge in edges:
            edge_list.append((edge['source'], edge['target']))
            
        self.graph = GraphData(num_nodes, edge_list)
        
        # Initial positions
        positions = {node['id']: Point(node['x'], node['y']) for node in nodes}
        
        # Bounds
        all_x = [node['x'] for node in nodes]
        all_y = [node['y'] for node in nodes]
        width = (max(all_x) - min(all_x)) + 20000 if all_x else 10000
        height = (max(all_y) - min(all_y)) + 20000 if all_y else 10000
        
        self.state = GridState(positions, width, height)
        
        # Prepare GPU Data
        self._prepare_gpu_data()

    def _prepare_gpu_data(self):
        # Edges: (E, 2) int32
        edges_np = np.array(self.graph.edges, dtype=np.int32)
        self.edges_gpu = cp.asarray(edges_np)
        
        # Positions: (N, 2) float32
        pos_np = np.zeros((self.graph.num_nodes, 2), dtype=np.float32)
        for i in range(self.graph.num_nodes):
            p = self.state.get_position(i)
            pos_np[i] = [p.x, p.y]
        self.positions_gpu = cp.asarray(pos_np)
        
        # Buffer for crossings
        self.edge_crossings_gpu = cp.zeros(len(self.graph.edges), dtype=np.int32)

    def _calculate_energy_gpu(self):
        num_edges = len(self.graph.edges)
        num_nodes = self.graph.num_nodes
        
        # 1. Calculate Crossings (Kernel)
        threads_per_block = 256
        blocks = (num_edges + threads_per_block - 1) // threads_per_block
        
        self.crossing_kernel(
            (blocks,), (threads_per_block,),
            (self.edges_gpu, self.positions_gpu, self.edge_crossings_gpu, num_edges)
        )
        
        # 2. Calculate Crossing Energy
        # energy = w_cross * sum(k^power)
        # We can do this with CuPy reductions
        crossing_energy = cp.sum(self.edge_crossings_gpu ** self.power) * self.w_cross
        
        # 3. Calculate Length Energy
        # length_sq = (x1-x2)^2 + (y1-y2)^2
        # Vectorized calculation
        src_indices = self.edges_gpu[:, 0]
        tgt_indices = self.edges_gpu[:, 1]
        
        p1 = self.positions_gpu[src_indices]
        p2 = self.positions_gpu[tgt_indices]
        
        diff = p1 - p2
        dist_sq = cp.sum(diff * diff, axis=1)
        length_energy = cp.sum(dist_sq) * self.w_len
        
        total_energy = crossing_energy + length_energy
        
        # Stats
        k = int(cp.max(self.edge_crossings_gpu))
        total_crossings = int(cp.sum(self.edge_crossings_gpu)) // 2 # Each crossing counted twice
        
        return float(total_energy), k, total_crossings, self.edge_crossings_gpu

    def solve(self, iterations=1000, initial_temp=50.0, cooling_rate=0.995, reheat_threshold=500, **kwargs):
        print("\n[START] CUDA solver starting...")
        
        # Warmup
        self._calculate_energy_gpu()
        cp.cuda.Stream.null.synchronize()
        
        temp = initial_temp
        current_energy, current_k, current_crossings, current_edge_crossings = self._calculate_energy_gpu()
        
        best_energy = current_energy
        best_positions = self.positions_gpu.copy()
        best_edge_crossings = current_edge_crossings.copy()
        
        accepted = 0
        start_time = time.time()
        
        # Main Loop
        # Note: For true speed, we should move the entire loop to GPU or batch moves.
        # But for now, we'll do CPU-driven SA with GPU evaluation.
        # This is still faster than CPU evaluation for large E.
        
        for i in range(iterations):
            # 1. Propose Move
            node_id = random.randint(0, self.graph.num_nodes - 1)
            old_x = float(self.positions_gpu[node_id, 0])
            old_y = float(self.positions_gpu[node_id, 1])
            
            radius = max(10, int(temp))
            new_x = old_x + random.randint(-radius, radius)
            new_y = old_y + random.randint(-radius, radius)
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                print(f"  [{i+1}/{iterations}] E={int(current_energy)}, Speed={speed:.1f} it/s")

        total_time = time.time() - start_time
        
        # Restore best
        self.positions_gpu = best_positions
        final_energy, final_k, final_crossings, _ = self._calculate_energy_gpu()
        
        # Convert to CPU for return
        final_positions_np = cp.asnumpy(self.positions_gpu)
        final_edge_crossings_np = cp.asnumpy(best_edge_crossings)
        
        nodes = []
        for i in range(self.graph.num_nodes):
            nodes.append({
                'id': i,
                'x': float(final_positions_np[i, 0]),
                'y': float(final_positions_np[i, 1])
            })
            
        return {
            'nodes': nodes,
            'energy': final_energy,
            'k': final_k,
            'total_crossings': final_crossings,
            'edge_crossings': final_edge_crossings_np,
            'iterations': iterations,
            'acceptance_rate': accepted / iterations,
            'time': total_time
        }

    def get_current_stats(self) -> Dict[str, Any]:
        energy, k, total, _ = self._calculate_energy_gpu()
        return {
            'energy': energy,
            'k': k,
            'total_crossings': total
        }

    def export_to_json(self, output_path: str):
        # Similar to other strategies
        pass
