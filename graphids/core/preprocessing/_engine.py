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

import structlog

import numpy as np
import torch
from torch_geometric.data import Data

from graphids.config import PREPROCESSING_DEFAULTS

from ._schema import IRSchema

log = structlog.get_logger()




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
        window_size: int = PREPROCESSING_DEFAULTS["window_size"],
        stride: int = PREPROCESSING_DEFAULTS["stride"],
    ):
        self.schema = schema
        self.window_size = window_size
        self.stride = stride
        from ._schema import build_node_manifest

        self.node_feature_count = build_node_manifest(schema.num_features).count

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
        from ._features import compute_edge_features, compute_node_features

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

        # Features (delegated to _features.py)
        edge_features = compute_edge_features(
            window,
            source,
            target,
            unique_edges,
            edge_counts,
            edge_inverse,
            nodes,
            edge_set=edge_set,
        )
        node_features = compute_node_features(
            window,
            nodes,
            source,
            self.node_feature_count,
            s.num_features,
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
        source_node_idx = np.array([node_to_idx[sv] for sv in source], dtype=np.int64)
        np.maximum.at(node_labels, source_node_idx, labels.astype(np.int64))
        data.node_y = torch.tensor(node_labels, dtype=torch.long)

        # Attack type metadata (graph-level and per-node)
        col_at = s.col_attack_type
        if col_at is not None:
            attack_types = window[:, col_at].astype(np.int64)
            # Graph-level: dominant non-normal attack type, or 0 if all normal
            attack_counts = np.bincount(attack_types)
            if len(attack_counts) > 1:
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
            np.maximum.at(node_attack_type, source_node_idx, attack_types)
            data.node_attack_type = torch.tensor(node_attack_type, dtype=torch.long)

        return data
