"""Leakage-safe train/validation graph indices."""

from __future__ import annotations

import math

import torch
from sklearn.model_selection import TimeSeriesSplit
from torch import Tensor
from torch_geometric.data import Data

from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
)


def _row_overlap_reach(window_size: int, stride: int) -> int:
    return 0 if stride >= window_size else max(0, math.ceil(window_size / stride) - 1)


def split_embargo_width(representation_cfg: GraphRepresentationCfg) -> int:
    if isinstance(representation_cfg, SnapshotRepresentationCfg):
        return _row_overlap_reach(representation_cfg.window_size, representation_cfg.stride)
    if isinstance(representation_cfg, SnapshotSequenceRepresentationCfg):
        return (
            (representation_cfg.sequence_length - 1) * representation_cfg.sequence_stride
            + _row_overlap_reach(representation_cfg.window_size, representation_cfg.stride)
        )
    raise TypeError(f"unsupported representation config: {type(representation_cfg)!r}")


def _num_graphs(data: Data, slices: dict[str, Tensor]) -> int:
    y = getattr(data, "y", None)
    return int(slices["x"].numel() - 1) if "x" in slices else int(y.numel()) if y is not None else 0


def _sliced_unique(data: Data, slices: dict[str, Tensor], key: str, graph_idx: int) -> tuple[int, ...]:
    start, end = int(slices[key][graph_idx]), int(slices[key][graph_idx + 1])
    return tuple(sorted({int(v) for v in getattr(data, key)[start:end].tolist()}))


def graph_touched_base_units(data: Data, slices: dict[str, Tensor]) -> list[tuple[int, ...]]:
    """Return base snapshot windows touched by each graph."""

    n_graphs = _num_graphs(data, slices)
    if hasattr(data, "node_snapshot_wid") and "node_snapshot_wid" in slices:
        return [_sliced_unique(data, slices, "node_snapshot_wid", idx) for idx in range(n_graphs)]
    if hasattr(data, "graph_wid"):
        return [(int(v),) for v in data.graph_wid[:n_graphs].tolist()]
    return [(idx,) for idx in range(n_graphs)]


def _dense(touched: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    ordinals = {unit: idx for idx, unit in enumerate(sorted({unit for units in touched for unit in units}))}
    return [tuple(sorted({ordinals[unit] for unit in units})) for units in touched]


def _tail_unit_split(n_units: int, val_fraction: float, embargo_width: int) -> tuple[set[int], set[int]]:
    n_val = int(n_units * val_fraction)
    if n_units == 0 or n_val <= 0:
        return set(range(n_units)), set()

    # TimeSeriesSplit cannot express a single split; small datasets use the same tail rule directly.
    if n_units > 2 * n_val + embargo_width:
        split = TimeSeriesSplit(n_splits=2, test_size=n_val, gap=embargo_width)
        train, val = list(split.split(range(n_units)))[-1]
        return set(train.tolist()), set(val.tolist())

    val_start = max(0, n_units - n_val)
    train_end = max(0, val_start - embargo_width)
    return set(range(train_end)), set(range(val_start, n_units))


def _contained_graph_indices(touched: list[tuple[int, ...]], allowed: set[int]) -> Tensor:
    idx = [
        graph_idx
        for graph_idx, units in enumerate(touched)
        if units and set(units).issubset(allowed)
    ]
    return torch.as_tensor(idx, dtype=torch.long)


def split_graph_indices(
    data: Data,
    slices: dict[str, Tensor],
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
) -> tuple[Tensor, Tensor]:
    """Return train/validation graph indices with no touched base-window overlap."""

    touched = _dense(graph_touched_base_units(data, slices))
    n_units = len({unit for units in touched for unit in units})
    embargo_width = split_embargo_width(representation_cfg)
    train_units, val_units = _tail_unit_split(n_units, val_fraction, embargo_width)
    return (
        _contained_graph_indices(touched, train_units),
        _contained_graph_indices(touched, val_units),
    )
