"""Declarative edge construction policies for graph preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EdgePolicy:
    """How to derive directed edges from windowed rows."""

    name: str
    src_col: str = "node_id"
    dst_col: str = "node_id"
    dst_shift: int = -1
    src_alias: str = "src"
    dst_alias: str = "dst"


def temporal_edge_policy(
    *,
    src_col: str = "node_id",
    dst_col: str = "node_id",
    dst_shift: int = -1,
) -> EdgePolicy:
    """Temporal adjacency policy: edge from row ``t`` to row ``t + dst_shift``."""
    return EdgePolicy(
        name="temporal_shift",
        src_col=src_col,
        dst_col=dst_col,
        dst_shift=dst_shift,
    )

