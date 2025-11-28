import numpy as np

def count_crossings(nodes_x, nodes_y, edges_source, edges_target):
    """
    Count crossings for each edge using vectorized NumPy operations.
    
    Args:
        nodes_x: Array of x coordinates
        nodes_y: Array of y coordinates
        edges_source: Array of source node ids
        edges_target: Array of target node ids
        
    Returns:
        edge_crossings: Array of crossing counts for each edge
        max_crossings: The maximum number of crossings for any single edge (k)
        total_crossings: Total number of crossings in the graph
    """
    num_edges = len(edges_source)
    if num_edges == 0:
        return np.zeros(0, dtype=np.int32), 0, 0
        
    # Coordinates of edge endpoints: (E, 2)
    P1 = np.stack([nodes_x[edges_source], nodes_y[edges_source]], axis=1)
    P2 = np.stack([nodes_x[edges_target], nodes_y[edges_target]], axis=1)
    
    # We need to compare every edge i with every edge j > i
    # To vectorize, we can broadcast.
    # But full E^2 broadcasting might be memory heavy if E is huge.
    # For E=1000, E^2 = 1M, which is fine (1M bools/floats is small).
    
    # Expand dims for broadcasting
    # A, B are shape (E, 1, 2)
    A = P1[:, np.newaxis, :]
    B = P2[:, np.newaxis, :]
    # C, D are shape (1, E, 2)
    C = P1[np.newaxis, :, :]
    D = P2[np.newaxis, :, :]
    
    # CCW function vectorized
    # ccw(A, B, C) = (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    # Result shape (E, E)
    
    def ccw_vec(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > \
               (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
               
    ccw_ACD = ccw_vec(A, C, D)
    ccw_BCD = ccw_vec(B, C, D)
    ccw_ABC = ccw_vec(A, B, C)
    ccw_ABD = ccw_vec(A, B, D)
    
    # Intersection condition: ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    intersect_mat = (ccw_ACD != ccw_BCD) & (ccw_ABC != ccw_ABD)
    
    # Handle shared endpoints
    # If endpoints match, it's not a crossing
    # A == C or A == D or B == C or B == D
    # We can check indices instead of coordinates for robustness
    # Sources: S (E, 1), Targets: T (E, 1)
    S = edges_source[:, np.newaxis]
    T = edges_target[:, np.newaxis]
    S_row = edges_source[np.newaxis, :]
    T_row = edges_target[np.newaxis, :]
    
    # Shared node check
    shared = (S == S_row) | (S == T_row) | (T == S_row) | (T == T_row)
    
    # Remove shared endpoint cases from intersection
    intersect_mat = intersect_mat & (~shared)
    
    # The matrix is symmetric and diagonal is False (self-intersection impossible due to shared check)
    # We only care about i < j (upper triangle)
    # But to get counts per edge, we can sum the whole row (since i intersects j implies j intersects i)
    # intersect_mat[i, j] is True if i intersects j.
    # Sum over columns gives total crossings for edge i.
    
    edge_crossings = np.sum(intersect_mat, axis=1)
    
    # Total crossings is sum(edge_crossings) / 2
    total_crossings = np.sum(edge_crossings) // 2
    max_crossings = np.max(edge_crossings) if num_edges > 0 else 0
    
    return edge_crossings.astype(np.int32), int(max_crossings), int(total_crossings)

def count_intersections_for_edges(target_indices, nodes_x, nodes_y, edges_source, edges_target):
    """
    Count crossings ONLY for the specified edges against all other edges.
    
    Args:
        target_indices: Indices of edges to check (e.g. incident to a moved node)
        nodes_x, nodes_y: Full coordinate arrays
        edges_source, edges_target: Full edge arrays
        
    Returns:
        local_crossings: Array of crossing counts for the target edges (aligned with target_indices)
    """
    num_targets = len(target_indices)
    num_edges = len(edges_source)
    if num_targets == 0 or num_edges == 0:
        return np.zeros(0, dtype=np.int32)
        
    # Target edges (T, 2)
    t_src = edges_source[target_indices]
    t_tgt = edges_target[target_indices]
    
    T1 = np.stack([nodes_x[t_src], nodes_y[t_src]], axis=1)
    T2 = np.stack([nodes_x[t_tgt], nodes_y[t_tgt]], axis=1)
    
    # All edges (E, 2)
    E1 = np.stack([nodes_x[edges_source], nodes_y[edges_source]], axis=1)
    E2 = np.stack([nodes_x[edges_target], nodes_y[edges_target]], axis=1)
    
    # Broadcast: Targets (T, 1, 2) vs All (1, E, 2)
    A = T1[:, np.newaxis, :]
    B = T2[:, np.newaxis, :]
    C = E1[np.newaxis, :, :]
    D = E2[np.newaxis, :, :]
    
    def ccw_vec(A, B, C):
        return (C[..., 1] - A[..., 1]) * (B[..., 0] - A[..., 0]) > \
               (B[..., 1] - A[..., 1]) * (C[..., 0] - A[..., 0])
               
    ccw_ACD = ccw_vec(A, C, D)
    ccw_BCD = ccw_vec(B, C, D)
    ccw_ABC = ccw_vec(A, B, C)
    ccw_ABD = ccw_vec(A, B, D)
    
    intersect_mat = (ccw_ACD != ccw_BCD) & (ccw_ABC != ccw_ABD)
    
    # Shared check
    # Targets: (T, 1), All: (1, E)
    S_tgt = t_src[:, np.newaxis]
    T_tgt = t_tgt[:, np.newaxis]
    S_all = edges_source[np.newaxis, :]
    T_all = edges_target[np.newaxis, :]
    
    shared = (S_tgt == S_all) | (S_tgt == T_all) | (T_tgt == S_all) | (T_tgt == T_all)
    
    intersect_mat = intersect_mat & (~shared)
    
    # Also exclude self-comparison if target index is in all edges (which it is)
    # But shared check handles this (shared endpoints), except for the edge itself where both endpoints match
    # Wait, if i == j, shared check handles it because S_tgt == S_all and T_tgt == T_all
    
    # Sum over all edges to get count for each target edge
    local_crossings = np.sum(intersect_mat, axis=1)
    
    return local_crossings.astype(np.int32)
