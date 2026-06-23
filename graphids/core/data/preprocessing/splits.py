"""Leakage-safe train/validation graph indices."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.data import Data

from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    SnapshotSequenceRepresentationCfg,
    representation_kind,
    representation_payload,
)

SPLIT_POLICY = "blocked_tail_v4"


@dataclass(frozen=True)
class SplitPlan:
    train_idx: Tensor
    val_idx: Tensor
    embargo_width: int
    digest: str
    meta: dict[str, object]
    audit: dict[str, int]

    def metadata(self) -> dict[str, object]:
        return dict(self.meta)


def split_policy_digest(
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
    seed: int,
    policy: str = SPLIT_POLICY,
) -> str:
    payload = {
        "policy": policy,
        "unit": "dense_base_window",
        "representation_kind": representation_kind(representation_cfg),
        "representation_cfg": representation_payload(representation_cfg),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


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


def _split_indices(touched: list[tuple[int, ...]], val_fraction: float, embargo_width: int) -> tuple[Tensor, Tensor]:
    n_units = len({unit for units in touched for unit in units})
    n_val = int(n_units * val_fraction)
    if n_units == 0 or n_val <= 0:
        return torch.arange(len(touched), dtype=torch.long), torch.empty(0, dtype=torch.long)

    val_start = n_units - n_val
    train_units = set(range(max(0, val_start - embargo_width)))
    val_units = set(range(val_start, n_units))

    def contained(units: tuple[int, ...], allowed: set[int]) -> bool:
        return bool(units) and set(units).issubset(allowed)

    train = [idx for idx, units in enumerate(touched) if contained(units, train_units)]
    val = [idx for idx, units in enumerate(touched) if contained(units, val_units)]
    return torch.as_tensor(train, dtype=torch.long), torch.as_tensor(val, dtype=torch.long)


def _row_intervals(data: Data, n_graphs: int) -> list[tuple[int, int]]:
    if hasattr(data, "window_start_row") and hasattr(data, "window_end_row"):
        return [
            (int(start), int(end))
            for start, end in zip(data.window_start_row[:n_graphs].tolist(), data.window_end_row[:n_graphs].tolist(), strict=False)
        ]
    if hasattr(data, "graph_wid"):
        return [(int(start), int(start) + 1) for start in data.graph_wid[:n_graphs].tolist()]
    return [(idx, idx + 1) for idx in range(n_graphs)]


def _interval_intersections(train: list[tuple[int, int]], val: list[tuple[int, int]]) -> int:
    count = 0
    val = sorted(val)
    for start, end in sorted(train):
        for val_start, val_end in val:
            if val_start >= end:
                break
            count += int(start < val_end)
    return count


def _audit(train_idx: Tensor, val_idx: Tensor, touched: list[tuple[int, ...]], intervals: list[tuple[int, int]]) -> dict[str, int]:
    train_graphs = set(train_idx.tolist())
    val_graphs = set(val_idx.tolist())
    train_units = {unit for idx in train_graphs for unit in touched[idx]}
    val_units = {unit for idx in val_graphs for unit in touched[idx]}
    return {
        "graph_index_overlap": len(train_graphs & val_graphs),
        "base_unit_overlap": len(train_units & val_units),
        "raw_interval_intersections": _interval_intersections(
            [intervals[idx] for idx in train_graphs],
            [intervals[idx] for idx in val_graphs],
        ),
    }


def audit_split_plan(plan: SplitPlan) -> dict[str, int]:
    return dict(plan.audit)


def build_blocked_split_plan(
    data: Data,
    slices: dict[str, Tensor],
    representation_cfg: GraphRepresentationCfg,
    *,
    val_fraction: float,
    seed: int,
    policy: str = SPLIT_POLICY,
) -> SplitPlan:
    touched = _dense(graph_touched_base_units(data, slices))
    embargo = split_embargo_width(representation_cfg)
    train_idx, val_idx = _split_indices(touched, val_fraction, embargo)
    digest = split_policy_digest(representation_cfg, val_fraction=val_fraction, seed=seed, policy=policy)
    meta = {
        "split_policy": policy,
        "split_unit": "dense_base_window",
        "split_embargo": embargo,
        "split_plan_digest": digest,
        "val_fraction": float(val_fraction),
        "seed": int(seed),
    }
    return SplitPlan(
        train_idx=train_idx,
        val_idx=val_idx,
        embargo_width=embargo,
        digest=digest,
        meta=meta,
        audit=_audit(train_idx, val_idx, touched, _row_intervals(data, len(touched))),
    )
