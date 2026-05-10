"""Temporal graph primitives for stream and sequence views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl
import torch

from graphids.core.data.preprocessing.edge_policy import (
    EdgePolicy,
    temporal_edge_policy,
)


@dataclass(frozen=True)
class TemporalGraphSpec:
    """How to turn ordered rows into a PyG ``TemporalData`` object."""

    edge_policy: EdgePolicy = field(default_factory=temporal_edge_policy)
    time_col: str = "timestamp"
    feature_cols: tuple[str, ...] = ()
    target_col: str = "attack"
    aux_label_cols: tuple[str, ...] = ("attack_type",)
    binary_target: bool = True


def _torchize(df: pl.DataFrame, cols: list[str], *, dtype: Any):
    return df.select(cols).fill_null(0).fill_nan(0).to_torch(dtype=dtype).squeeze(-1)


def build_temporal_data(df: pl.DataFrame, spec: TemporalGraphSpec) -> Any:
    """Build a PyG ``TemporalData`` stream from ordered rows."""
    from torch_geometric.data import TemporalData

    rows = df.sort(spec.time_col).with_columns(
        pl.col(spec.edge_policy.src_col).alias("src"),
        pl.col(spec.edge_policy.dst_col)
        .shift(-spec.edge_policy.dst_shift)
        .alias("dst"),
    )
    rows = rows.filter(pl.col("dst").is_not_null())

    src = _torchize(rows, ["src"], dtype=pl.Int64)
    dst = _torchize(rows, ["dst"], dtype=pl.Int64)
    t = rows.select(spec.time_col).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32).squeeze(-1)

    kwargs: dict[str, Any] = {"src": src, "dst": dst, "t": t}
    if spec.feature_cols:
        kwargs["msg"] = rows.select(list(spec.feature_cols)).fill_null(0).fill_nan(0).to_torch(dtype=pl.Float32)

    target = rows.select(spec.target_col).fill_null(0).fill_nan(0).to_torch(dtype=pl.Int64).squeeze(-1)
    kwargs["y"] = (target > 0).to(torch.int64) if spec.binary_target else target

    for col in spec.aux_label_cols:
        if col in rows.columns:
            kwargs[col] = rows.select(col).fill_null(0).fill_nan(0).to_torch(dtype=pl.Int64).squeeze(-1)

    return TemporalData(**kwargs)


def temporal_len(data: Any) -> int:
    """Return the number of events if the object exposes it."""
    return int(getattr(data, "num_events", len(data)))
