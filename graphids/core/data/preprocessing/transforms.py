"""Reusable graph-transform expressions for dataset schemas."""

from __future__ import annotations

import polars as pl

TOPOLOGY_NODE_FEATURE_COLS: tuple[str, ...] = (
    "clustering_coeff",
    "in_degree",
    "out_degree",
)

TOPOLOGY_NODE_PLACEHOLDER_EXPRS: list[pl.Expr] = [
    pl.lit(0.0).alias(c) for c in TOPOLOGY_NODE_FEATURE_COLS
]

