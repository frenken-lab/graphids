"""Domain-agnostic graph construction engine.

Receives an IR DataFrame (conforming to ``IRSchema``) and produces
PyTorch Geometric ``Data`` objects via sliding-window graph construction.

The engine knows nothing about CAN buses, network flows, or any other
domain — it only operates on the standardized column layout.

Phase 3.3: Edge and node feature computation is fully vectorized using
``np.unique(return_inverse=True)`` + ``np.add.at`` scatter operations,
replacing the O(E×W) and O(N×W) Python loops.

Phase 4: Extended node features (26-D) with per-byte std, payload entropy,
payload change rate, skewness/kurtosis, split-half ratio, and clustering
coefficient.  Aljabri et al. (2025) validated entropy features via SHAP.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch_geometric.data import Data

from graphids.config.constants import (
    DEFAULT_STRIDE,
    DEFAULT_WINDOW_SIZE,
    EDGE_FEATURE_COUNT,
)

from .schema import IRSchema

log = logging.getLogger(__name__)


class GraphEngine:
    """Converts IR DataFrames into PyG graph objects.

    Parameters
    ----------
    schema : IRSchema
        Describes the column layout of incoming DataFrames.
    window_size : int
        Number of rows per sliding window.
    stride : int
        Step size between consecutive windows.
    """

    def __init__(
        self,
        schema: IRSchema,
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
    ):
        self.schema = schema
        self.window_size = window_size
        self.stride = stride
        # 26-D node features:
        #   entity_id(1) + byte_means(num_features) + byte_stds(num_features)
        #   + payload_entropy(1) + change_rate_mean(1) + change_rate_max(1)
        #   + skewness(1) + kurtosis(1) + clustering_coeff(1) + split_half(1)
        #   + occurrence_count(1) + last_position(1)
        self.node_feature_count = 1 + schema.num_features * 2 + 7 + 2

    def create_graphs(self, ir_df) -> list[Data]:
        """Transform an IR DataFrame into a list of PyG Data objects.

        Parameters
        ----------
        ir_df : pd.DataFrame
            DataFrame conforming to ``self.schema``.

        Returns
        -------
        list[Data]
            One graph per sliding window.
        """
        data_array = ir_df.to_numpy()
        n = len(data_array)
        ws, st = self.window_size, self.stride
        num_windows = max(1, (n - ws) // st + 1)

        graphs = []
        for w in range(num_windows):
            start = w * st
            window = data_array[start : start + ws]
            graphs.append(self._window_to_graph(window))

        return graphs

    # ------------------------------------------------------------------
    # Internal: per-window graph construction
    # ------------------------------------------------------------------

    def _window_to_graph(self, window: np.ndarray) -> Data:
        """Build a single PyG Data from a numpy window slice."""
        s = self.schema

        source = window[:, s.col_source]
        target = window[:, s.col_target]
        labels = window[:, s.col_label]

        # Unique edges, counts, and inverse mapping for scatter ops
        edges = np.column_stack((source, target))
        unique_edges, edge_inverse, edge_counts = np.unique(
            edges,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        # Node mapping (dense re-indexing)
        nodes = np.unique(np.concatenate((source, target)))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)

        edge_index = np.array([[node_to_idx[src], node_to_idx[tgt]] for src, tgt in unique_edges]).T
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Build edge set for bidirectionality + clustering coefficient
        edge_set = set(map(tuple, unique_edges))

        # Features (vectorized)
        edge_features = self._compute_edge_features_vectorized(
            window,
            source,
            target,
            unique_edges,
            edge_counts,
            edge_inverse,
            nodes,
            edge_set=edge_set,
        )
        node_features = self._compute_node_features_vectorized(
            window,
            nodes,
            source,
            edge_index_np=edge_index.numpy(),
            num_nodes=num_nodes,
        )

        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        x = torch.tensor(node_features, dtype=torch.float)
        label_value = 1 if np.any(labels == 1) else 0
        y = torch.tensor(label_value, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # ID sequence entropy (graph-level attribute)
        entity_ids = window[:, 0].astype(np.int64)
        id_counts = np.bincount(entity_ids)
        id_counts = id_counts[id_counts > 0]
        id_probs = id_counts / id_counts.sum()
        data.id_entropy = torch.tensor(
            float(-np.sum(id_probs * np.log2(id_probs + 1e-12))),
            dtype=torch.float,
        )

        # Per-node binary labels (attack=1, normal=0) via scatter-max
        node_labels = np.zeros(num_nodes, dtype=np.int64)
        for row_idx in range(len(source)):
            nidx = node_to_idx[source[row_idx]]
            if labels[row_idx] == 1:
                node_labels[nidx] = 1
        data.node_y = torch.tensor(node_labels, dtype=torch.long)

        # Attack type metadata (graph-level and per-node)
        col_at = s.col_attack_type
        if col_at is not None:
            attack_types = window[:, col_at].astype(np.int64)
            # Graph-level: dominant non-normal attack type, or 0 if all normal
            attack_counts = np.bincount(attack_types)
            if len(attack_counts) > 1:
                # Zero out normal (code 0) to find dominant attack type
                attack_counts_no_normal = attack_counts.copy()
                attack_counts_no_normal[0] = 0
                if attack_counts_no_normal.sum() > 0:
                    data.attack_type = torch.tensor(
                        int(np.argmax(attack_counts_no_normal)), dtype=torch.long
                    )
                else:
                    data.attack_type = torch.tensor(0, dtype=torch.long)
            else:
                data.attack_type = torch.tensor(0, dtype=torch.long)

            # Per-node attack type via scatter (take max code per node)
            node_attack_type = np.zeros(num_nodes, dtype=np.int64)
            for row_idx in range(len(source)):
                nidx = node_to_idx[source[row_idx]]
                if attack_types[row_idx] > node_attack_type[nidx]:
                    node_attack_type[nidx] = attack_types[row_idx]
            data.node_attack_type = torch.tensor(node_attack_type, dtype=torch.long)

        return data

    def _compute_edge_features_vectorized(
        self,
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
        """Compute 11-D edge features using vectorized operations.

        Replaces the O(E×W) Python loop with scatter-based aggregation.

        Feature layout (matches legacy ``EDGE_FEATURE_COUNT=11``):
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
        W = len(window)
        E = len(unique_edges)
        features = np.zeros((E, EDGE_FEATURE_COUNT), dtype=np.float32)

        # [0-1] Frequency features (already vectorized)
        features[:, 0] = edge_counts
        features[:, 1] = edge_counts / W

        # --- Temporal features via scatter ---
        # edge_inverse[row] = which unique edge this row belongs to
        positions = np.arange(W, dtype=np.float64)

        # First and last occurrence per edge group
        first_pos = np.full(E, W, dtype=np.float64)  # init to max
        last_pos = np.full(E, -1.0, dtype=np.float64)  # init to min
        np.minimum.at(first_pos, edge_inverse, positions)
        np.maximum.at(last_pos, edge_inverse, positions)

        # [5-7] Temporal position features (normalized)
        valid_mask = edge_counts > 0
        features[valid_mask, 5] = first_pos[valid_mask] / W
        features[valid_mask, 6] = last_pos[valid_mask] / W
        features[valid_mask, 7] = (last_pos[valid_mask] - first_pos[valid_mask]) / W

        # [2-4] Interval statistics: mean, std, regularity
        # For edges with count > 1, compute intervals between sorted positions
        multi_mask = edge_counts > 1
        if np.any(multi_mask):
            # Sort positions by edge group for interval computation
            sort_idx = np.lexsort((positions, edge_inverse))
            sorted_groups = edge_inverse[sort_idx]
            sorted_positions = positions[sort_idx]

            # Compute intervals: diff between consecutive same-group positions
            # A pair (i, i+1) belongs to the same group when sorted_groups match
            same_group = sorted_groups[:-1] == sorted_groups[1:]
            intervals = np.diff(sorted_positions)

            # Only keep intervals within the same edge group
            valid_intervals = intervals[same_group]
            valid_groups = sorted_groups[:-1][same_group]

            if len(valid_intervals) > 0:
                # Sum and sum-of-squares per edge group for mean/std
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
                # Variance = E[X^2] - E[X]^2, then sqrt for std
                variance = np.zeros(E, dtype=np.float64)
                variance[has_intervals] = (
                    interval_sq_sum[has_intervals] / interval_count[has_intervals]
                    - mean_interval[has_intervals] ** 2
                )
                # Clamp negative variance from float precision
                variance = np.maximum(variance, 0.0)
                std_interval[has_intervals] = np.sqrt(variance[has_intervals])

                features[has_intervals, 2] = mean_interval[has_intervals]
                features[has_intervals, 3] = std_interval[has_intervals]
                # Regularity: 1/(1+std), with std=0 → regularity=1
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
        # Map unique_edges src/tgt to node indices for degree lookup
        all_uniq = np.unique(all_nodes_arr)
        n2i = {n: i for i, n in enumerate(all_uniq)}
        src_deg = np.array([degree[n2i[s]] for s, _ in unique_edges], dtype=np.float32)
        tgt_deg = np.array([degree[n2i[t]] for _, t in unique_edges], dtype=np.float32)
        features[:, 9] = src_deg * tgt_deg
        features[:, 10] = src_deg / np.maximum(tgt_deg, 1e-8)

        return features

    def _compute_node_features_vectorized(
        self,
        window: np.ndarray,
        nodes: np.ndarray,
        source: np.ndarray,
        *,
        edge_index_np: np.ndarray | None = None,
        num_nodes: int | None = None,
    ) -> np.ndarray:
        """Compute 26-D node features using vectorized scatter operations.

        Feature layout (for CAN bus with num_features=8, total 26):
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

        For non-CAN domains with different num_features, the layout scales:
            entity_id(1) + means(n) + stds(n) + 7 scalar features + count + position
        """
        s = self.schema
        n_feat = s.num_features
        N = len(nodes)
        if num_nodes is None:
            num_nodes = N
        W = len(source)
        feat_end = 1 + n_feat  # columns 0..feat_end (exclusive)

        node_features = np.zeros((N, self.node_feature_count), dtype=np.float32)

        # Map source values to node indices
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        source_node_idx = np.array([node_to_idx[sv] for sv in source], dtype=np.int64)

        # --- Scatter-sum for means, sum-of-squares for stds ---
        row_features = window[:, :feat_end].astype(np.float64)
        feature_sums = np.zeros((N, feat_end), dtype=np.float64)
        feature_sq_sums = np.zeros((N, n_feat), dtype=np.float64)
        # For skewness (sum of cubes) and kurtosis (sum of fourth powers)
        feature_cube_sums = np.zeros((N, n_feat), dtype=np.float64)
        feature_quad_sums = np.zeros((N, n_feat), dtype=np.float64)
        occurrence_counts = np.zeros(N, dtype=np.float64)

        # Scatter: accumulate per-node statistics
        for col in range(feat_end):
            np.add.at(feature_sums[:, col], source_node_idx, row_features[:, col])
        payload_cols = row_features[:, 1:feat_end]  # feature_0..feature_N
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
            node_features[has_data, 1 + col] = (
                feature_sums[has_data, 1 + col] / counts_safe[has_data]
            )

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

        # --- [1+2*n_feat] Payload entropy per node ---
        entropy_idx = 1 + 2 * n_feat
        # Collect byte value distributions per node. Payload bytes are normalized [0,1],
        # so we quantize to 256 bins for entropy computation.
        byte_counts_per_node = np.zeros((N, 256), dtype=np.float64)
        for col in range(n_feat):
            byte_vals = np.clip((payload_cols[:, col] * 255).astype(np.int32), 0, 255)
            for row_idx in range(W):
                nidx = source_node_idx[row_idx]
                byte_counts_per_node[nidx, byte_vals[row_idx]] += 1
        for i in range(N):
            total = byte_counts_per_node[i].sum()
            if total > 0:
                probs = byte_counts_per_node[i] / total
                probs = probs[probs > 0]
                node_features[i, entropy_idx] = -np.sum(probs * np.log2(probs))

        # --- [entropy_idx+1:entropy_idx+3] Payload change rate (mean/max abs change) ---
        change_mean_idx = entropy_idx + 1
        change_max_idx = entropy_idx + 2
        positions = np.arange(W, dtype=np.float64)
        # Sort by (node, position) to get consecutive appearances per node
        sort_order = np.lexsort((positions, source_node_idx))
        sorted_nodes = source_node_idx[sort_order]
        sorted_payload = payload_cols[sort_order]
        # Find consecutive same-node pairs
        same_node = sorted_nodes[:-1] == sorted_nodes[1:]
        if np.any(same_node):
            payload_diff = np.abs(sorted_payload[1:] - sorted_payload[:-1])
            # Mean abs change across all bytes per transition
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

        # --- [change_max_idx+1:change_max_idx+3] Skewness and Kurtosis (scalar avg) ---
        skew_idx = change_max_idx + 1
        kurt_idx = change_max_idx + 2
        # Skewness = E[(X-μ)³] / σ³, Kurtosis = E[(X-μ)⁴] / σ⁴ - 3 (excess)
        # Using raw moments: skew = (E[X³] - 3μσ² - μ³) / σ³
        for i in range(N):
            if not has_data[i]:
                continue
            cnt = counts_safe[i]
            per_byte_skew = np.zeros(n_feat, dtype=np.float64)
            per_byte_kurt = np.zeros(n_feat, dtype=np.float64)
            valid_bytes = 0
            for col in range(n_feat):
                mean_val = feature_sums[i, 1 + col] / cnt
                mean_sq = feature_sq_sums[i, col] / cnt
                var = max(mean_sq - mean_val**2, 0.0)
                std = np.sqrt(var)
                if std < 1e-8:
                    continue
                mean_cube = feature_cube_sums[i, col] / cnt
                mean_quad = feature_quad_sums[i, col] / cnt
                # Central moments from raw moments
                m3 = mean_cube - 3 * mean_val * mean_sq + 2 * mean_val**3
                m4 = (
                    mean_quad
                    - 4 * mean_val * mean_cube
                    + 6 * mean_val**2 * mean_sq
                    - 3 * mean_val**4
                )
                per_byte_skew[col] = m3 / (std**3)
                per_byte_kurt[col] = m4 / (std**4) - 3.0  # excess kurtosis
                valid_bytes += 1
            if valid_bytes > 0:
                node_features[i, skew_idx] = per_byte_skew.sum() / valid_bytes
                node_features[i, kurt_idx] = per_byte_kurt.sum() / valid_bytes

        # --- [kurt_idx+1] Clustering coefficient ---
        clust_idx = kurt_idx + 1
        if edge_index_np is not None and edge_index_np.shape[1] > 0:
            # Build adjacency sets per node (undirected for triangle counting)
            adj = [set() for _ in range(num_nodes)]
            for e in range(edge_index_np.shape[1]):
                u, v = int(edge_index_np[0, e]), int(edge_index_np[1, e])
                adj[u].add(v)
                adj[v].add(u)
            for i in range(num_nodes):
                neighbors = adj[i]
                k = len(neighbors)
                if k < 2:
                    continue
                triangles = 0
                neighbor_list = list(neighbors)
                for ni in range(k):
                    for nj in range(ni + 1, k):
                        if neighbor_list[nj] in adj[neighbor_list[ni]]:
                            triangles += 1
                node_features[i, clust_idx] = 2.0 * triangles / (k * (k - 1))

        # --- [clust_idx+1] Split-half ratio ---
        split_idx = clust_idx + 1
        half_w = W / 2.0
        # First-half and second-half mean payload per node
        first_half_sum = np.zeros(N, dtype=np.float64)
        first_half_count = np.zeros(N, dtype=np.float64)
        second_half_sum = np.zeros(N, dtype=np.float64)
        second_half_count = np.zeros(N, dtype=np.float64)
        # Average across all payload bytes per row
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
            # Ratio with small epsilon to avoid division by zero
            node_features[both_halves, split_idx] = first_mean / (second_mean + 1e-8)

        # --- [-1] Last temporal position per node (vectorized) ---
        last_pos = np.full(N, -1.0, dtype=np.float64)
        np.maximum.at(last_pos, source_node_idx, positions)
        has_pos = last_pos >= 0
        node_features[has_pos, -1] = last_pos[has_pos] / max(W - 1, 1)

        # --- [-2] Normalized occurrence counts ---
        c_min, c_max = occurrence_counts.min(), occurrence_counts.max()
        if c_max > c_min:
            node_features[:, -2] = ((occurrence_counts - c_min) / (c_max - c_min)).astype(
                np.float32
            )
        else:
            node_features[:, -2] = occurrence_counts.astype(np.float32)

        return node_features
