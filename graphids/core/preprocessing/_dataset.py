"""Graph dataset wrapper with collated tensor storage.

Uses PyG's collation to store all graphs as concatenated tensors + a slices
dictionary.  This reduces mmap region count from N_graphs × tensors_per_graph
to just ~10 total tensors, enabling num_workers > 0 in DataLoader even for
large datasets (previously blocked by vm.max_map_count ≈ 65530).

Migration note (v3.0.0): Old caches stored ``list[Data]`` (one tensor storage
per attribute per graph).  New caches store ``(data_dict, slices_dict)`` via
``save_collated()`` / ``load_collated()``.  Bumping PREPROCESSING_VERSION
triggers automatic rebuild.
"""

from __future__ import annotations

import structlog
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate

log = structlog.get_logger()


def GraphDataset(data_list: list[Data]) -> CollatedGraphDataset:
    """Create a CollatedGraphDataset from a list of Data objects.

    Drop-in replacement for the old list-based GraphDataset class.
    Callers that did ``GraphDataset(graphs)`` continue to work unchanged.
    """
    try:
        data, slices = InMemoryDataset.collate(data_list)
    except RuntimeError as e:
        raise ValueError(str(e)) from e
    ds = CollatedGraphDataset(data, slices)
    ds.validate_consistency()
    return ds


def save_collated(graphs: list[Data], path: Path) -> dict:
    """Collate a list of Data objects and save as a single .pt file.

    Returns the slices dict (useful for writing metadata without reloading).
    """
    data, slices = InMemoryDataset.collate(graphs)
    torch.save({"data": data.to_dict(), "slices": slices}, path, pickle_protocol=4)
    return slices


def load_collated(path: Path, mmap: bool = True) -> CollatedGraphDataset:
    """Load a collated .pt file into a CollatedGraphDataset."""
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=mmap,
        )
    except TypeError:
        # PyTorch < 2.1: no mmap support
        log.warning("mmap not supported (PyTorch < 2.1), using standard load")
        payload = torch.load(path, map_location="cpu", weights_only=False)

    # Support both new collated format and legacy list format
    if isinstance(payload, dict) and "data" in payload and "slices" in payload:
        collated = Data.from_dict(payload["data"])
        slices = payload["slices"]
        return CollatedGraphDataset(collated, slices)
    else:
        # Legacy format: list[Data] — collate on the fly
        log.warning("Legacy list[Data] cache detected. Will be rebuilt on next preprocessing.")
        graphs = payload
        if hasattr(graphs, "data_list"):
            graphs = graphs.data_list
        if not isinstance(graphs, list):
            graphs = list(graphs)
        data, slices = InMemoryDataset.collate(graphs)
        return CollatedGraphDataset(data, slices)


class CollatedGraphDataset(torch.utils.data.Dataset):
    """Dataset backed by collated tensors — zero-copy __getitem__ via views.

    Instead of storing N separate Data objects (N × T tensor storages),
    all attributes are concatenated into shared tensors with a slices dict
    recording per-graph boundaries.  ``separate()`` returns views into the
    shared storage, so no data is copied on access.
    """

    def __init__(self, data: Data, slices: dict[str, torch.Tensor]):
        super().__init__()
        self._data = data
        self._slices = slices
        # Length from any slice tensor (all have len = num_graphs + 1)
        self._len = next(iter(slices.values())).size(0) - 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Data:
        return separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self._slices,
            decrement=False,
        )

    @property
    def tensor_storage_count(self) -> int:
        """Number of mmap-relevant tensor storages (for num_workers safety check)."""
        data_tensors = sum(1 for v in self._data.values() if isinstance(v, torch.Tensor))
        slice_tensors = sum(1 for v in self._slices.values() if isinstance(v, torch.Tensor))
        return data_tensors + slice_tensors

    def validate_consistency(self) -> None:
        """Validate feature dimensions are consistent (spot-check first and last)."""
        if self._len == 0:
            return
        first = self[0]
        last = self[self._len - 1]
        for g, label in [(first, "first"), (last, "last")]:
            if g.x is not None and first.x is not None and g.x.size(1) != first.x.size(1):
                raise ValueError(
                    f"Graph {label} has inconsistent node features: "
                    f"expected {first.x.size(1)}, got {g.x.size(1)}"
                )

    def get_stats(self) -> dict[str, int | float]:
        """Compute dataset statistics directly from collated tensors (no per-graph iteration)."""
        if self._len == 0:
            return {"num_graphs": 0}

        # Node counts from x slices
        x_slices = self._slices.get("x")
        if x_slices is not None:
            node_counts = (x_slices[1:] - x_slices[:-1]).numpy()
        else:
            node_counts = np.zeros(self._len)

        # Edge counts from edge_index slices
        ei_slices = self._slices.get("edge_index")
        if ei_slices is not None:
            edge_counts = (ei_slices[1:] - ei_slices[:-1]).numpy()
        else:
            edge_counts = np.zeros(self._len)

        # Labels from collated y
        y = self._data.y
        if y is not None:
            labels = y.numpy().flatten()
        else:
            labels = np.zeros(self._len)

        # Node feature dim
        node_features = self._data.x.size(1) if self._data.x is not None else 0
        edge_features = self._data.edge_attr.size(1) if self._data.edge_attr is not None else 0

        return {
            "num_graphs": int(self._len),
            "avg_nodes": float(np.mean(node_counts)),
            "std_nodes": float(np.std(node_counts)),
            "avg_edges": float(np.mean(edge_counts)),
            "std_edges": float(np.std(edge_counts)),
            "min_nodes": int(np.min(node_counts)),
            "max_nodes": int(np.max(node_counts)),
            "min_edges": int(np.min(edge_counts)),
            "max_edges": int(np.max(edge_counts)),
            "normal_graphs": int(np.sum(labels == 0)),
            "attack_graphs": int(np.sum(labels == 1)),
            "node_features": node_features,
            "edge_features": edge_features,
        }
