"""Node and edge feature computation for graph construction.

Extracted from engine.py. Uses scipy.stats.entropy for Shannon entropy
and networkx.clustering for clustering coefficients (both are C-optimized
and already transitive deps of PyG).

Feature layouts:
    Node (26-D for CAN bus with 8 payload bytes):
        [0]        entity_id mean
        [1:9]      per-byte mean of payload bytes
        [9:17]     per-byte std of payload bytes
        [17]       payload entropy (Shannon, over byte values)
        [18]       payload change rate — mean abs change
        [19]       payload change rate — max abs change
        [20]       skewness (scalar avg across bytes)
        [21]       kurtosis (scalar avg across bytes)
        [22]       clustering coefficient (local)
        [23]       split-half ratio (first-half mean / second-half mean)
        [24]       normalized occurrence count
        [25]       last temporal position (normalized)

    Edge (11-D):
        [0]  raw count
        [1]  frequency (count / window_length)
        [2]  mean interval between occurrences
        [3]  std interval
        [4]  regularity  1/(1+std)
        [5]  first occurrence position (normalized)
        [6]  last occurrence position (normalized)
        [7]  temporal span (last - first)
        [8]  bidirectionality flag
        [9]  degree product (src_deg * tgt_deg)
        [10] degree ratio (src_deg / tgt_deg)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as _scipy_entropy

from graphids.config import EDGE_FEATURE_COUNT


def compute_edge_features(
    window: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    unique_edges: np.ndarray,
    edge_counts: np.ndarray,
    edge_inverse: np.ndarray,
    nodes: np.ndarray,
    *,
    edge_set: set[tuple] | None = None,
) -> np.ndarray:
    """Compute 11-D edge features using vectorized scatter operations."""
    W = len(window)
    E = len(unique_edges)
    features = np.zeros((E, EDGE_FEATURE_COUNT), dtype=np.float32)

    # [0-1] Frequency features
    features[:, 0] = edge_counts
    features[:, 1] = edge_counts / W

    # --- Temporal features via scatter ---
    positions = np.arange(W, dtype=np.float64)

    # First and last occurrence per edge group
    first_pos = np.full(E, W, dtype=np.float64)
    last_pos = np.full(E, -1.0, dtype=np.float64)
    np.minimum.at(first_pos, edge_inverse, positions)
    np.maximum.at(last_pos, edge_inverse, positions)

    # [5-7] Temporal position features (normalized)
    valid_mask = edge_counts > 0
    features[valid_mask, 5] = first_pos[valid_mask] / W
    features[valid_mask, 6] = last_pos[valid_mask] / W
    features[valid_mask, 7] = (last_pos[valid_mask] - first_pos[valid_mask]) / W

    # [2-4] Interval statistics: mean, std, regularity
    multi_mask = edge_counts > 1
    if np.any(multi_mask):
        sort_idx = np.lexsort((positions, edge_inverse))
        sorted_groups = edge_inverse[sort_idx]
        sorted_positions = positions[sort_idx]

        same_group = sorted_groups[:-1] == sorted_groups[1:]
        intervals = np.diff(sorted_positions)

        valid_intervals = intervals[same_group]
        valid_groups = sorted_groups[:-1][same_group]

        if len(valid_intervals) > 0:
            interval_sum = np.zeros(E, dtype=np.float64)
            interval_sq_sum = np.zeros(E, dtype=np.float64)
            interval_count = np.zeros(E, dtype=np.float64)

            np.add.at(interval_sum, valid_groups, valid_intervals)
            np.add.at(interval_sq_sum, valid_groups, valid_intervals**2)
            np.add.at(interval_count, valid_groups, 1.0)

            has_intervals = interval_count > 0
            mean_interval = np.zeros(E, dtype=np.float64)
            std_interval = np.zeros(E, dtype=np.float64)

            mean_interval[has_intervals] = (
                interval_sum[has_intervals] / interval_count[has_intervals]
            )
            variance = np.zeros(E, dtype=np.float64)
            variance[has_intervals] = (
                interval_sq_sum[has_intervals] / interval_count[has_intervals]
                - mean_interval[has_intervals] ** 2
            )
            variance = np.maximum(variance, 0.0)
            std_interval[has_intervals] = np.sqrt(variance[has_intervals])

            features[has_intervals, 2] = mean_interval[has_intervals]
            features[has_intervals, 3] = std_interval[has_intervals]
            reg = np.ones(E, dtype=np.float64)
            nonzero_std = std_interval > 0
            reg[nonzero_std] = 1.0 / (1.0 + std_interval[nonzero_std])
            features[multi_mask, 4] = reg[multi_mask]

    # [8] Bidirectionality: check if reverse edge exists
    if edge_set is None:
        edge_set = set(map(tuple, unique_edges))
    features[:, 8] = np.array(
        [float((tgt, src) in edge_set) for src, tgt in unique_edges],
        dtype=np.float32,
    )

    # [9-10] Degree features
    all_nodes_arr = np.concatenate([source, target])
    _, node_inv = np.unique(all_nodes_arr, return_inverse=True)
    degree = np.bincount(node_inv)
    all_uniq = np.unique(all_nodes_arr)
    n2i = {n: i for i, n in enumerate(all_uniq)}
    src_deg = np.array([degree[n2i[s]] for s, _ in unique_edges], dtype=np.float32)
    tgt_deg = np.array([degree[n2i[t]] for _, t in unique_edges], dtype=np.float32)
    features[:, 9] = src_deg * tgt_deg
    features[:, 10] = src_deg / np.maximum(tgt_deg, 1e-8)

    return features


def compute_node_features(
    window: np.ndarray,
    nodes: np.ndarray,
    source: np.ndarray,
    node_feature_count: int,
    num_features: int,
    *,
    edge_index_np: np.ndarray | None = None,
    num_nodes: int | None = None,
) -> np.ndarray:
    """Compute 26-D node features using vectorized scatter + scipy/networkx."""
    N = len(nodes)
    if num_nodes is None:
        num_nodes = N
    W = len(source)
    n_feat = num_features
    feat_end = 1 + n_feat

    node_features = np.zeros((N, node_feature_count), dtype=np.float32)

    # Map source values to node indices
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    source_node_idx = np.array([node_to_idx[sv] for sv in source], dtype=np.int64)

    # --- Scatter-sum for means, sum-of-squares for stds ---
    row_features = window[:, :feat_end].astype(np.float64)
    feature_sums = np.zeros((N, feat_end), dtype=np.float64)
    feature_sq_sums = np.zeros((N, n_feat), dtype=np.float64)
    feature_cube_sums = np.zeros((N, n_feat), dtype=np.float64)
    feature_quad_sums = np.zeros((N, n_feat), dtype=np.float64)
    occurrence_counts = np.zeros(N, dtype=np.float64)

    # Scatter: accumulate per-node statistics
    for col in range(feat_end):
        np.add.at(feature_sums[:, col], source_node_idx, row_features[:, col])
    payload_cols = row_features[:, 1:feat_end]
    for col in range(n_feat):
        np.add.at(feature_sq_sums[:, col], source_node_idx, payload_cols[:, col] ** 2)
        np.add.at(feature_cube_sums[:, col], source_node_idx, payload_cols[:, col] ** 3)
        np.add.at(feature_quad_sums[:, col], source_node_idx, payload_cols[:, col] ** 4)
    np.add.at(occurrence_counts, source_node_idx, 1.0)

    has_data = occurrence_counts > 0
    counts_safe = np.where(has_data, occurrence_counts, 1.0)

    # [0] entity_id mean
    node_features[has_data, 0] = feature_sums[has_data, 0] / counts_safe[has_data]

    # [1:1+n_feat] per-byte means
    for col in range(n_feat):
        node_features[has_data, 1 + col] = feature_sums[has_data, 1 + col] / counts_safe[has_data]

    # [1+n_feat:1+2*n_feat] per-byte stds: sqrt(E[X²] - E[X]²)
    std_offset = 1 + n_feat
    for col in range(n_feat):
        mean_val = feature_sums[has_data, 1 + col] / counts_safe[has_data]
        mean_sq = feature_sq_sums[has_data, col] / counts_safe[has_data]
        variance = np.maximum(mean_sq - mean_val**2, 0.0)
        node_features[has_data, std_offset + col] = np.sqrt(variance)

    # For target-only nodes (no source occurrences), set entity_id directly
    target_only = ~has_data
    if np.any(target_only):
        node_features[target_only, 0] = nodes[target_only]

    # --- [1+2*n_feat] Payload entropy per node (scipy) ---
    entropy_idx = 1 + 2 * n_feat
    # Vectorized byte histogram: quantize payload to 256 bins, scatter into per-node counts
    byte_counts_per_node = np.zeros((N, 256), dtype=np.float64)
    for col in range(n_feat):
        byte_vals = np.clip((payload_cols[:, col] * 255).astype(np.int32), 0, 255)
        np.add.at(byte_counts_per_node, (source_node_idx, byte_vals), 1)
    # scipy.stats.entropy with base=2 computes Shannon entropy per row
    for i in range(N):
        total = byte_counts_per_node[i].sum()
        if total > 0:
            node_features[i, entropy_idx] = _scipy_entropy(byte_counts_per_node[i], base=2)

    # --- [entropy_idx+1:entropy_idx+3] Payload change rate (mean/max abs change) ---
    change_mean_idx = entropy_idx + 1
    change_max_idx = entropy_idx + 2
    positions = np.arange(W, dtype=np.float64)
    sort_order = np.lexsort((positions, source_node_idx))
    sorted_nodes = source_node_idx[sort_order]
    sorted_payload = payload_cols[sort_order]
    same_node = sorted_nodes[:-1] == sorted_nodes[1:]
    if np.any(same_node):
        payload_diff = np.abs(sorted_payload[1:] - sorted_payload[:-1])
        per_transition_mean = payload_diff[same_node].mean(axis=1)
        per_transition_max = payload_diff[same_node].max(axis=1)
        groups = sorted_nodes[:-1][same_node]

        change_sum = np.zeros(N, dtype=np.float64)
        change_max = np.zeros(N, dtype=np.float64)
        change_count = np.zeros(N, dtype=np.float64)
        np.add.at(change_sum, groups, per_transition_mean)
        np.maximum.at(change_max, groups, per_transition_max)
        np.add.at(change_count, groups, 1.0)

        has_changes = change_count > 0
        node_features[has_changes, change_mean_idx] = (
            change_sum[has_changes] / change_count[has_changes]
        )
        node_features[has_changes, change_max_idx] = change_max[has_changes]

    # --- Skewness and Kurtosis (vectorized across all nodes at once) ---
    skew_idx = change_max_idx + 1
    kurt_idx = change_max_idx + 2
    _compute_skew_kurtosis(
        node_features,
        feature_sums,
        feature_sq_sums,
        feature_cube_sums,
        feature_quad_sums,
        counts_safe,
        has_data,
        n_feat,
        skew_idx,
        kurt_idx,
    )

    # --- Clustering coefficient (networkx, C-optimized) ---
    clust_idx = kurt_idx + 1
    if edge_index_np is not None and edge_index_np.shape[1] > 0:
        _compute_clustering_coefficients(node_features, edge_index_np, num_nodes, clust_idx)

    # --- Split-half ratio ---
    split_idx = clust_idx + 1
    half_w = W / 2.0
    first_half_sum = np.zeros(N, dtype=np.float64)
    first_half_count = np.zeros(N, dtype=np.float64)
    second_half_sum = np.zeros(N, dtype=np.float64)
    second_half_count = np.zeros(N, dtype=np.float64)
    row_payload_mean = payload_cols.mean(axis=1)
    first_mask = positions < half_w
    second_mask = ~first_mask
    np.add.at(first_half_sum, source_node_idx[first_mask], row_payload_mean[first_mask])
    np.add.at(first_half_count, source_node_idx[first_mask], 1.0)
    np.add.at(second_half_sum, source_node_idx[second_mask], row_payload_mean[second_mask])
    np.add.at(second_half_count, source_node_idx[second_mask], 1.0)
    both_halves = (first_half_count > 0) & (second_half_count > 0)
    if np.any(both_halves):
        first_mean = first_half_sum[both_halves] / first_half_count[both_halves]
        second_mean = second_half_sum[both_halves] / second_half_count[both_halves]
        node_features[both_halves, split_idx] = first_mean / (second_mean + 1e-8)

    # --- Last temporal position per node ---
    last_pos = np.full(N, -1.0, dtype=np.float64)
    np.maximum.at(last_pos, source_node_idx, positions)
    has_pos = last_pos >= 0
    node_features[has_pos, -1] = last_pos[has_pos] / max(W - 1, 1)

    # --- Normalized occurrence counts ---
    c_min, c_max = occurrence_counts.min(), occurrence_counts.max()
    if c_max > c_min:
        node_features[:, -2] = ((occurrence_counts - c_min) / (c_max - c_min)).astype(np.float32)
    else:
        node_features[:, -2] = occurrence_counts.astype(np.float32)

    # --- Normalize features to [0, 1] for sigmoid-based VGAE decoder ---
    node_features[:, entropy_idx] = np.clip(node_features[:, entropy_idx] / 8.0, 0.0, 1.0)
    node_features[:, skew_idx] = (node_features[:, skew_idx] + 10.0) / 20.0
    node_features[:, kurt_idx] = (node_features[:, kurt_idx] + 10.0) / 20.0
    node_features[:, split_idx] = np.clip(node_features[:, split_idx] / 10.0, 0.0, 1.0)

    return node_features


def _compute_skew_kurtosis(
    node_features: np.ndarray,
    feature_sums: np.ndarray,
    feature_sq_sums: np.ndarray,
    feature_cube_sums: np.ndarray,
    feature_quad_sums: np.ndarray,
    counts_safe: np.ndarray,
    has_data: np.ndarray,
    n_feat: int,
    skew_idx: int,
    kurt_idx: int,
) -> None:
    """Vectorized skewness/kurtosis across all nodes at once."""
    # Extract only nodes with data
    idx = np.where(has_data)[0]
    if len(idx) == 0:
        return

    cnt = counts_safe[idx]  # (M,)
    per_byte_skew_sum = np.zeros(len(idx), dtype=np.float64)
    per_byte_kurt_sum = np.zeros(len(idx), dtype=np.float64)
    valid_bytes_count = np.zeros(len(idx), dtype=np.float64)

    for col in range(n_feat):
        mean_val = feature_sums[idx, 1 + col] / cnt
        mean_sq = feature_sq_sums[idx, col] / cnt
        var = np.maximum(mean_sq - mean_val**2, 0.0)
        std = np.sqrt(var)

        valid = std >= 1e-8
        if not np.any(valid):
            continue

        mean_cube = feature_cube_sums[idx[valid], col] / cnt[valid]
        mean_quad = feature_quad_sums[idx[valid], col] / cnt[valid]
        mv = mean_val[valid]
        ms = mean_sq[valid]
        s = std[valid]

        # Central moments from raw moments
        m3 = mean_cube - 3 * mv * ms + 2 * mv**3
        m4 = mean_quad - 4 * mv * mean_cube + 6 * mv**2 * ms - 3 * mv**4

        skew = np.clip(m3 / (s**3), -10.0, 10.0)
        kurt = np.clip(m4 / (s**4) - 3.0, -10.0, 10.0)

        per_byte_skew_sum[valid] += skew
        per_byte_kurt_sum[valid] += kurt
        valid_bytes_count[valid] += 1.0

    has_valid = valid_bytes_count > 0
    node_features[idx[has_valid], skew_idx] = (
        per_byte_skew_sum[has_valid] / valid_bytes_count[has_valid]
    )
    node_features[idx[has_valid], kurt_idx] = (
        per_byte_kurt_sum[has_valid] / valid_bytes_count[has_valid]
    )


def _compute_clustering_coefficients(
    node_features: np.ndarray,
    edge_index_np: np.ndarray,
    num_nodes: int,
    clust_idx: int,
) -> None:
    """Compute local clustering coefficients using networkx (C-optimized)."""
    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index_np.T.tolist()
    G.add_edges_from(edges)

    clustering = nx.clustering(G)
    for node_id, cc in clustering.items():
        if node_id < len(node_features):
            node_features[node_id, clust_idx] = cc
