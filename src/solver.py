import numpy as np
import math
import random
import networkx as nx
try:
    from scorer import count_crossings, count_intersections_for_edges
except ImportError:
    from src.scorer import count_crossings, count_intersections_for_edges

class SimulatedAnnealingSolver:
    def __init__(self, nodes, edges, width, height):
        """
        Initialize the solver.
        """
        self.width = width
        self.height = height
        self.edges_data = edges
        self.edges_source = np.array([e['source'] for e in edges], dtype=np.int32)
        self.edges_target = np.array([e['target'] for e in edges], dtype=np.int32)
        self.num_nodes = len(nodes)
        self.num_edges = len(edges)
        
        # Build Adjacency Map: Node ID -> List of Edge Indices
        self.node_to_edges = [[] for _ in range(self.num_nodes)]
        for i, e in enumerate(edges):
            u, v = e['source'], e['target']
            if u < self.num_nodes: self.node_to_edges[u].append(i)
            if v < self.num_nodes: self.node_to_edges[v].append(i)
            
        # Initialize node positions
        self.nodes_x = np.zeros(self.num_nodes, dtype=np.float64)
        self.nodes_y = np.zeros(self.num_nodes, dtype=np.float64)
        
        # Load input positions first
        for n in nodes:
            nid = n['id']
            if nid < self.num_nodes:
                self.nodes_x[nid] = n['x']
                self.nodes_y[nid] = n['y']
                
        # Smart Initialization Check
        # If the bounding box of the input is very small compared to the canvas,
        # OR if we want to force a better layout, we use NetworkX.
        
        # Calculate bounding box of input
        min_x, max_x = np.min(self.nodes_x), np.max(self.nodes_x)
        min_y, max_y = np.min(self.nodes_y), np.max(self.nodes_y)
        w_span = max_x - min_x
        h_span = max_y - min_y
        
        # Heuristic: If graph uses less than 20% of the canvas in either dimension, it's "clumped".
        # Or if it's the default 0-100 range on a 1M canvas.
        is_clumped = (w_span < self.width * 0.2) or (h_span < self.height * 0.2)
        
        if is_clumped:
            # Use NetworkX Spring Layout
            G = nx.Graph()
            G.add_nodes_from(range(self.num_nodes))
            G.add_edges_from([(e['source'], e['target']) for e in edges])
            
            # Compute layout (scale=1 gives roughly [-1, 1] or [0, 1])
            pos = nx.spring_layout(G, iterations=50)
            
            # Extract and Scale to full canvas with padding
            px = np.array([pos[i][0] for i in range(self.num_nodes)])
            py = np.array([pos[i][1] for i in range(self.num_nodes)])
            
            # Normalize to 0..1
            p_min_x, p_max_x = np.min(px), np.max(px)
            p_min_y, p_max_y = np.min(py), np.max(py)
            
            if p_max_x > p_min_x:
                px = (px - p_min_x) / (p_max_x - p_min_x)
            if p_max_y > p_min_y:
                py = (py - p_min_y) / (p_max_y - p_min_y)
                
            # Scale to canvas with 5% padding
            padding_x = self.width * 0.05
            padding_y = self.height * 0.05
            usable_w = self.width * 0.9
            usable_h = self.height * 0.9
            
            self.nodes_x = px * usable_w + padding_x
            self.nodes_y = py * usable_h + padding_y
            
        # Pre-calculate full state
        self.edge_crossings, _, _ = count_crossings(self.nodes_x, self.nodes_y, self.edges_source, self.edges_target)
        
    def current_state(self):
        return self.nodes_x.copy(), self.nodes_y.copy()
        
    def energy(self, nodes_x=None, nodes_y=None):
        # If coordinates are provided, compute from scratch (slow, for tests/validation)
        if nodes_x is None: nodes_x = self.nodes_x
        if nodes_y is None: nodes_y = self.nodes_y
        
        # 1. Crossings
        if nodes_x is not self.nodes_x or nodes_y is not self.nodes_y:
             _, k, total = count_crossings(nodes_x, nodes_y, self.edges_source, self.edges_target)
        else:
             k = np.max(self.edge_crossings) if self.num_edges > 0 else 0
             total = np.sum(self.edge_crossings) // 2
             
        # Dynamic K Weight: Ensure it dominates spring/repulsion
        k_weight = max(10000, self.width * 0.1)
        crossing_energy = k * k_weight + total
             
        # 2. Repulsion (Dynamic Radius)
        # Radius = 15% of width
        radius = self.width * 0.15
        
        coords = np.stack([nodes_x, nodes_y], axis=1)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        repulsion_mask = dists < radius
        # Penalty: (Radius - dist)
        repulsion_vals = radius - dists
        repulsion_energy = np.sum(repulsion_vals[repulsion_mask]) / 2 
        
        # 3. Spring Attraction (Edge Lengths)
        u_x = nodes_x[self.edges_source]
        u_y = nodes_y[self.edges_source]
        v_x = nodes_x[self.edges_target]
        v_y = nodes_y[self.edges_target]
        
        edge_lengths = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2)
        spring_energy = np.sum(edge_lengths) * 0.05
        
        return crossing_energy + repulsion_energy + spring_energy

    def solve(self, iterations=1000, temp=10.0, cooling_rate=0.995):
        # Constants
        radius = self.width * 0.15
        k_weight = max(10000, self.width * 0.1)
        
        # Initial State
        current_x = self.nodes_x.copy()
        current_y = self.nodes_y.copy()
        
        # Initial Components
        # Crossings
        current_k = np.max(self.edge_crossings) if self.num_edges > 0 else 0
        current_total_crossings = np.sum(self.edge_crossings) // 2
        current_crossing_energy = current_k * k_weight + current_total_crossings
        
        # Repulsion
        coords = np.stack([current_x, current_y], axis=1)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        rep_mask = dists < radius
        current_repulsion_energy = np.sum((radius - dists)[rep_mask]) / 2
        
        # Spring
        u_x = current_x[self.edges_source]
        u_y = current_y[self.edges_source]
        v_x = current_x[self.edges_target]
        v_y = current_y[self.edges_target]
        edge_lengths = np.sqrt((u_x - v_x)**2 + (u_y - v_y)**2)
        current_spring_energy = np.sum(edge_lengths) * 0.05
        
        current_total_energy = current_crossing_energy + current_repulsion_energy + current_spring_energy
        
        best_x = current_x.copy()
        best_y = current_y.copy()
        best_energy = current_total_energy
        
        steps_since_improvement = 0
        
        for i in range(iterations):
            # Propose a move
            node_idx = np.random.randint(0, self.num_nodes)
            
            old_x = self.nodes_x[node_idx]
            old_y = self.nodes_y[node_idx]
            
            delta = temp
            new_x = old_x + np.random.uniform(-delta, delta)
            new_y = old_y + np.random.uniform(-delta, delta)
            
            new_x = max(0, min(self.width, new_x))
            new_y = max(0, min(self.height, new_y))
            
            # --- Incremental Updates ---
            
            # 1. Repulsion Delta (O(N))
            d_old = np.sqrt((self.nodes_x - old_x)**2 + (self.nodes_y - old_y)**2)
            d_old[node_idx] = np.inf 
            rep_old = np.sum(radius - d_old[d_old < radius])
            
            d_new = np.sqrt((self.nodes_x - new_x)**2 + (self.nodes_y - new_y)**2)
            d_new[node_idx] = np.inf
            rep_new = np.sum(radius - d_new[d_new < radius])
            
            delta_repulsion = rep_new - rep_old
            
            # 2. Spring Delta (O(d))
            incident_edge_indices = self.node_to_edges[node_idx]
            delta_spring = 0
            
            if incident_edge_indices:
                incident_indices = np.array(incident_edge_indices, dtype=np.int32)
                e_src = self.edges_source[incident_indices]
                e_tgt = self.edges_target[incident_indices]
                
                is_source = (e_src == node_idx)
                neighbor_indices = np.where(is_source, e_tgt, e_src)
                
                nbr_x = self.nodes_x[neighbor_indices]
                nbr_y = self.nodes_y[neighbor_indices]
                
                l_old = np.sqrt((old_x - nbr_x)**2 + (old_y - nbr_y)**2)
                l_new = np.sqrt((new_x - nbr_x)**2 + (new_y - nbr_y)**2)
                
                delta_spring = np.sum(l_new - l_old) * 0.05
            
            # 3. Crossing Delta (O(d*E))
            if not incident_edge_indices:
                delta_crossing_energy = 0
                temp_crossings = self.edge_crossings # No change
            else:
                # Helper to find intersections (Local)
                def get_intersections(indices, x, y):
                    t_src = self.edges_source[indices]
                    t_tgt = self.edges_target[indices]
                    
                    T1 = np.stack([x[t_src], y[t_src]], axis=1)
                    T2 = np.stack([x[t_tgt], y[t_tgt]], axis=1)
                    
                    E1 = np.stack([x[self.edges_source], y[self.edges_source]], axis=1)
                    E2 = np.stack([x[self.edges_target], y[self.edges_target]], axis=1)
                    
                    A = T1[:, np.newaxis, :]
                    B = T2[:, np.newaxis, :]
                    C = E1[np.newaxis, :, :]
                    D = E2[np.newaxis, :, :]
                    
                    def ccw(A, B, C):
                        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > \
                               (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
                    
                    mat = (ccw(A, C, D) != ccw(B, C, D)) & (ccw(A, B, C) != ccw(A, B, D))
                    
                    S_tgt = t_src[:, np.newaxis]
                    T_tgt = t_tgt[:, np.newaxis]
                    S_all = self.edges_source[np.newaxis, :]
                    T_all = self.edges_target[np.newaxis, :]
                    shared = (S_tgt == S_all) | (S_tgt == T_all) | (T_tgt == S_all) | (T_tgt == T_all)
                    
                    mat = mat & (~shared)
                    return mat

                # OLD STATE
                old_mat = get_intersections(incident_indices, self.nodes_x, self.nodes_y)
                
                # MOVE (Temporarily update self.nodes_x/y for the check)
                self.nodes_x[node_idx] = new_x
                self.nodes_y[node_idx] = new_y
                
                # NEW STATE
                new_mat = get_intersections(incident_indices, self.nodes_x, self.nodes_y)
                
                # Revert for now
                self.nodes_x[node_idx] = old_x
                self.nodes_y[node_idx] = old_y
                
                # Calculate Crossing Change
                temp_crossings = self.edge_crossings.copy()
                
                rows, cols = np.where(old_mat)
                edges_to_decrement = np.concatenate((incident_indices[rows], cols))
                decrements = np.bincount(edges_to_decrement, minlength=self.num_edges)
                temp_crossings -= decrements
                
                rows, cols = np.where(new_mat)
                edges_to_increment = np.concatenate((incident_indices[rows], cols))
                increments = np.bincount(edges_to_increment, minlength=self.num_edges)
                temp_crossings += increments
                
                new_k = np.max(temp_crossings) if self.num_edges > 0 else 0
                new_total = np.sum(temp_crossings) // 2
                new_crossing_energy = new_k * k_weight + new_total
                
                delta_crossing_energy = new_crossing_energy - current_crossing_energy

            # Total Energy Change
            total_delta = delta_crossing_energy + delta_repulsion + delta_spring
            new_energy_val = current_total_energy + total_delta
            
            # Metropolis
            diff = current_total_energy - new_energy_val # Positive if improvement
            prob = 1.0
            if diff < 0:
                try:
                    prob = math.exp(diff / temp)
                except OverflowError:
                    prob = 0.0
            
            if diff > 0 or np.random.random() < prob:
                # Accept
                current_total_energy = new_energy_val
                current_crossing_energy += delta_crossing_energy
                current_repulsion_energy += delta_repulsion
                current_spring_energy += delta_spring
                
                self.nodes_x[node_idx] = new_x
                self.nodes_y[node_idx] = new_y
                self.edge_crossings = temp_crossings
                
                steps_since_improvement = 0
                
                if new_energy_val < best_energy:
                    best_energy = new_energy_val
                    best_x = self.nodes_x.copy()
                    best_y = self.nodes_y.copy()
            else:
                # Reject
                steps_since_improvement += 1
                
            # Re-heat Logic
            if steps_since_improvement > 500:
                temp = max(temp * 2.0, self.width * 0.1) # Boost
                steps_since_improvement = 0
                
            temp *= cooling_rate
            
        self.nodes_x = best_x
        self.nodes_y = best_y
        return best_x, best_y, best_energy, temp
