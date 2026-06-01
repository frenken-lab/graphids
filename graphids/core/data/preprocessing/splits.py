"""Representation-aware train/validation split planning."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch_geometric.data import Data

from graphids.core.data.preprocessing.representations import (
    EntityRepresentationCfg,
    GraphRepresentationCfg,
    MultiScaleRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
    representation_payload,
)

SPLIT_POLICY = "blocked_tail_v4"


@dataclass(frozen=True)
class SplitPlan:
    """Graph indices and underlying base units for a train/validation split."""

    policy: str
    unit: str
    train_idx: Tensor
    val_idx: Tensor
    train_units: tuple[int, ...]
    val_units: tuple[int, ...]
    embargo_units: tuple[int, ...]
    train_intervals: tuple[tuple[int, int], ...]
    val_intervals: tuple[tuple[int, int], ...]
    embargo_width: int
    source_boundary_violations: int
    val_fraction: float
    seed: int
    digest: str

    def metadata(self) -> dict[str, Any]:
        return {
            "split_policy": self.policy,
            "split_unit": self.unit,
            "split_embargo": self.embargo_width,
            "split_plan_digest": self.digest,
            "val_fraction": self.val_fraction,
            "seed": self.seed,
            "num_train_units": len(self.train_units),
            "num_val_units": len(self.val_units),
            "num_embargo_units": len(self.embargo_units),
            "num_train_intervals": len(self.train_intervals),
            "num_val_intervals": len(self.val_intervals),
            "source_boundary_violations": self.source_boundary_violations,
        }


def _digest(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def split_policy_payload(
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
    seed: int,
    policy: str = SPLIT_POLICY,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "unit": "dense_base_window",
        "representation_kind": representation_kind(representation_cfg),
        "representation_cfg": representation_payload(representation_cfg),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
    }


def split_policy_digest(
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
    seed: int,
    policy: str = SPLIT_POLICY,
) -> str:
    return _digest(
        split_policy_payload(
            representation_cfg,
            val_fraction=val_fraction,
            seed=seed,
            policy=policy,
        )
    )


def _row_overlap_reach(window_size: int, stride: int) -> int:
    if stride >= window_size:
        return 0
    return max(0, math.ceil(window_size / stride) - 1)


def split_embargo_width(representation_cfg: GraphRepresentationCfg) -> int:
    """Embargo width in dense base-window ordinals."""

    if isinstance(representation_cfg, SnapshotRepresentationCfg):
        return _row_overlap_reach(representation_cfg.window_size, representation_cfg.stride)
    if isinstance(representation_cfg, SnapshotSequenceRepresentationCfg):
        sequence_reach = (representation_cfg.sequence_length - 1) * representation_cfg.sequence_stride
        return sequence_reach + _row_overlap_reach(
            representation_cfg.window_size,
            representation_cfg.stride,
        )
    if isinstance(representation_cfg, MultiScaleRepresentationCfg):
        return max(
            _row_overlap_reach(window_size, representation_cfg.stride)
            for window_size in representation_cfg.window_sizes
        )
    if isinstance(representation_cfg, EntityRepresentationCfg):
        return representation_cfg.history_window_size + representation_cfg.future_window_size
    if isinstance(representation_cfg, TemporalRepresentationCfg):
        return int(representation_cfg.history_messages or 0)
    raise TypeError(f"unsupported representation config: {type(representation_cfg)!r}")


def _num_graphs(data: Data, slices: dict[str, Tensor]) -> int:
    if "x" in slices:
        return int(slices["x"].numel() - 1)
    y = getattr(data, "y", None)
    if y is not None:
        return int(y.numel())
    return 0


def _graph_attr_values(data: Data, slices: dict[str, Tensor], key: str, graph_idx: int) -> tuple[int, ...]:
    values = getattr(data, key)
    offsets = slices[key]
    start = int(offsets[graph_idx])
    end = int(offsets[graph_idx + 1])
    if end <= start:
        return ()
    return tuple(sorted({int(v) for v in values[start:end].tolist()}))


def graph_touched_base_units(data: Data, slices: dict[str, Tensor]) -> list[tuple[int, ...]]:
    """Return underlying dense-ish base units touched by each materialized graph.

    For sequence graphs, this uses per-node ``node_snapshot_wid`` values. For
    simple snapshot graphs, this uses graph-level ``graph_wid``. The values are
    later remapped through their sorted order to dense base-window ordinals.
    """

    n_graphs = _num_graphs(data, slices)
    if n_graphs == 0:
        return []
    if hasattr(data, "node_snapshot_wid") and "node_snapshot_wid" in slices:
        return [
            _graph_attr_values(data, slices, "node_snapshot_wid", graph_idx)
            for graph_idx in range(n_graphs)
        ]
    if hasattr(data, "graph_wid"):
        return [(int(v),) for v in data.graph_wid[:n_graphs].tolist()]
    return [(graph_idx,) for graph_idx in range(n_graphs)]


def _graph_row_intervals(data: Data, n_graphs: int) -> list[tuple[int, int]]:
    if hasattr(data, "window_start_row") and hasattr(data, "window_end_row"):
        starts = data.window_start_row[:n_graphs].tolist()
        ends = data.window_end_row[:n_graphs].tolist()
        return [(int(start), int(end)) for start, end in zip(starts, ends, strict=False)]
    if hasattr(data, "graph_wid"):
        starts = data.graph_wid[:n_graphs].tolist()
        return [(int(start), int(start) + 1) for start in starts]
    return [(idx, idx + 1) for idx in range(n_graphs)]


def _graph_source_boundary_violations(data: Data, n_graphs: int) -> list[bool]:
    out = [False] * n_graphs
    for key in ("source_dir_n_unique", "source_file_n_unique"):
        values = getattr(data, key, None)
        if values is None:
            continue
        for idx, value in enumerate(values[:n_graphs].tolist()):
            out[idx] = out[idx] or int(value) > 1
    return out


def _graph_labels(data: Data, n_graphs: int) -> list[int] | None:
    y = getattr(data, "y", None)
    if y is None:
        return None
    return [int(value) for value in y[:n_graphs].tolist()]


def _indices_for_unit_set(
    touched: list[tuple[int, ...]],
    units: set[int],
) -> list[int]:
    return [
        graph_idx
        for graph_idx, graph_units in enumerate(touched)
        if graph_units and set(graph_units).issubset(units)
    ]


def _choose_validation_start(
    touched: list[tuple[int, ...]],
    labels: list[int] | None,
    *,
    n_units: int,
    n_val: int,
) -> int:
    tail_start = max(0, n_units - n_val)
    if labels is None or n_val <= 0:
        return tail_start

    total_pos = sum(1 for label in labels if label == 1)
    total_neg = len(labels) - total_pos
    if total_pos == 0 or total_neg == 0:
        return tail_start

    max_start = max(0, n_units - n_val)
    pos_diff = [0] * (max_start + 2)
    neg_diff = [0] * (max_start + 2)
    for graph_units, label in zip(touched, labels, strict=False):
        if not graph_units:
            continue
        first_unit = min(graph_units)
        last_unit = max(graph_units)
        start_lo = max(0, last_unit - n_val + 1)
        start_hi = min(first_unit, max_start)
        if start_lo > start_hi:
            continue
        diff = pos_diff if label == 1 else neg_diff
        diff[start_lo] += 1
        diff[start_hi + 1] -= 1

    pos_counts: list[int] = []
    neg_counts: list[int] = []
    pos = 0
    neg = 0
    for idx in range(max_start + 1):
        pos += pos_diff[idx]
        neg += neg_diff[idx]
        pos_counts.append(pos)
        neg_counts.append(neg)

    min_pos = max(1, int(total_pos * 0.10))
    min_pos = min(min_pos, total_pos)
    for start in range(max_start, -1, -1):
        if pos_counts[start] >= min_pos and neg_counts[start] > 0:
            return start

    best_start = tail_start
    best_key = (-1, -1, -1)
    for start, (pos_count, neg_count) in enumerate(zip(pos_counts, neg_counts, strict=False)):
        key = (pos_count, int(neg_count > 0), start)
        if key > best_key:
            best_key = key
            best_start = start
    return best_start


def intervals_intersect(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def audit_split_plan(plan: SplitPlan) -> dict[str, int]:
    raw_interval_intersections = 0
    for train_interval in plan.train_intervals:
        for val_interval in plan.val_intervals:
            if intervals_intersect(train_interval, val_interval):
                raw_interval_intersections += 1
    return {
        "graph_index_overlap": len(set(plan.train_idx.tolist()) & set(plan.val_idx.tolist())),
        "base_unit_overlap": len(set(plan.train_units) & set(plan.val_units)),
        "raw_interval_intersections": raw_interval_intersections,
        "source_boundary_violations": plan.source_boundary_violations,
    }


def build_blocked_split_plan(
    data: Data,
    slices: dict[str, Tensor],
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
    seed: int,
    policy: str = SPLIT_POLICY,
) -> SplitPlan:
    touched_raw = graph_touched_base_units(data, slices)
    n_graphs = len(touched_raw)
    intervals = _graph_row_intervals(data, n_graphs)
    boundary_violations = _graph_source_boundary_violations(data, n_graphs)
    raw_units = sorted({unit for graph_units in touched_raw for unit in graph_units})
    ordinal = {unit: i for i, unit in enumerate(raw_units)}
    touched = [
        tuple(sorted({ordinal[unit] for unit in graph_units}))
        for graph_units in touched_raw
    ]

    n_units = len(raw_units)
    n_val = int(n_units * val_fraction)
    embargo_width = split_embargo_width(representation_cfg)
    if n_val <= 0:
        train_units = tuple(range(n_units))
        val_units: tuple[int, ...] = ()
        embargo_units: tuple[int, ...] = ()
    else:
        val_start = _choose_validation_start(
            touched,
            _graph_labels(data, n_graphs),
            n_units=n_units,
            n_val=n_val,
        )
        val_end = min(n_units, val_start + n_val)
        train_end = max(0, val_start - embargo_width)
        train_units = tuple(range(train_end))
        embargo_units = tuple(range(train_end, val_start))
        val_units = tuple(range(val_start, val_end))

    train_set = set(train_units)
    val_set = set(val_units)

    train_idx = [
        graph_idx
        for graph_idx, graph_units in enumerate(touched)
        if graph_units and set(graph_units).issubset(train_set)
    ]
    val_idx = [
        graph_idx
        for graph_idx, graph_units in enumerate(touched)
        if graph_units and set(graph_units).issubset(val_set)
    ]

    digest = split_policy_digest(
        representation_cfg,
        val_fraction=val_fraction,
        seed=seed,
        policy=policy,
    )
    return SplitPlan(
        policy=policy,
        unit="dense_base_window",
        train_idx=torch.as_tensor(train_idx, dtype=torch.long),
        val_idx=torch.as_tensor(val_idx, dtype=torch.long),
        train_units=train_units,
        val_units=val_units,
        embargo_units=embargo_units,
        train_intervals=tuple(intervals[idx] for idx in train_idx),
        val_intervals=tuple(intervals[idx] for idx in val_idx),
        embargo_width=embargo_width,
        source_boundary_violations=sum(boundary_violations),
        val_fraction=float(val_fraction),
        seed=int(seed),
        digest=digest,
    )
