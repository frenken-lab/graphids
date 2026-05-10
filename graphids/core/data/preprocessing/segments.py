"""Segment primitives for alternate dataset views.

The current graph pipeline treats one sliding window as one graph sample.
These primitives let us describe richer sample shapes without baking that
choice into the CAN adapter or graph builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import Literal

import polars as pl


@dataclass(frozen=True)
class WindowSegmentCfg:
    """One fixed window of rows."""

    window_size: int
    stride: int


@dataclass(frozen=True)
class SequenceSegmentCfg:
    """A sequence of ordered windows from the same raw stream."""

    window_size: int
    stride: int
    sequence_length: int = 4
    sequence_stride: int = 1


@dataclass(frozen=True)
class MultiScaleSegmentCfg:
    """Parallel windows at multiple temporal scales."""

    window_sizes: tuple[int, ...]
    stride: int


@dataclass(frozen=True)
class EntitySegmentCfg:
    """A segment centered on one arbitration ID or message family."""

    anchor_column: str = "node_id"
    anchor_value: str | int | None = None
    history_window_size: int = 100
    future_window_size: int = 0


SegmentCfg = WindowSegmentCfg | SequenceSegmentCfg | MultiScaleSegmentCfg | EntitySegmentCfg


@dataclass(frozen=True)
class GraphSegmentPlan:
    """Resolved sample-shape plan for a dataset view."""

    kind: Literal["window", "sequence", "multi_scale", "entity"]
    cfg: SegmentCfg


@dataclass(frozen=True)
class WindowedRows:
    """Rows plus derived window metadata for snapshot-style segments."""

    rows: pl.DataFrame
    n_rows: int
    n_windows: int
    max_wid: int


class Segmenter(Protocol):
    """Primitive that turns raw rows into a shaped sample view."""

    def segment(self, df: pl.DataFrame) -> WindowedRows:
        raise NotImplementedError


@dataclass(frozen=True)
class WindowSegmenter:
    """Default snapshot segmenter: one fixed sliding window per graph."""

    window_size: int
    stride: int

    def segment(self, df: pl.DataFrame) -> WindowedRows:
        rows = df.with_row_index("_row").with_columns(pl.col("_row").cast(pl.Int64))
        n_rows = len(rows)
        n_windows = max(0, (n_rows - self.window_size) // self.stride + 1)
        max_wid = (n_windows - 1) * self.stride
        rows = rows.with_columns(
            (pl.col("_row") % self.window_size < (self.window_size // 2)).alias(
                "_first_half"
            )
        )
        return WindowedRows(rows=rows, n_rows=n_rows, n_windows=n_windows, max_wid=max_wid)


def segment_kind(cfg: SegmentCfg) -> str:
    """Stable human-readable label for logging and config routing."""
    if isinstance(cfg, WindowSegmentCfg):
        return "window"
    if isinstance(cfg, SequenceSegmentCfg):
        return "sequence"
    if isinstance(cfg, MultiScaleSegmentCfg):
        return "multi_scale"
    if isinstance(cfg, EntitySegmentCfg):
        return "entity"
    raise TypeError(f"unsupported segment config: {type(cfg)!r}")


def segment_plan(cfg: SegmentCfg) -> GraphSegmentPlan:
    """Wrap a segment config with its stable kind label."""
    return GraphSegmentPlan(kind=segment_kind(cfg), cfg=cfg)
