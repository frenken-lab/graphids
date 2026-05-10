"""View primitives for turning raw CAN/event data into graph samples.

The current pipeline is snapshot-window based. This module drafts a small
set of view configs so we can support alternate dataset lenses without
hard-coding them into the schema adapter or the graph pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class _ViewCfg:
    """Marker base for future typed view configs."""


@dataclass(frozen=True)
class SnapshotViewCfg(_ViewCfg):
    """One fixed graph per sliding window."""

    window_size: int = 100
    stride: int = 100


@dataclass(frozen=True)
class SnapshotSequenceViewCfg(_ViewCfg):
    """An ordered sequence of snapshot graphs."""

    window_size: int = 100
    stride: int = 100
    sequence_length: int = 4
    sequence_stride: int = 1


@dataclass(frozen=True)
class MultiScaleViewCfg(_ViewCfg):
    """Parallel snapshot views at multiple window sizes."""

    window_sizes: tuple[int, ...] = (50, 100, 200)
    stride: int = 100


@dataclass(frozen=True)
class EventChunkViewCfg(_ViewCfg):
    """Chunk raw events by message count or duration."""

    message_count: int | None = 200
    duration_ms: float | None = None
    overlap: float = 0.0


@dataclass(frozen=True)
class RollingStreamViewCfg(_ViewCfg):
    """Online/streaming view with bounded history."""

    history_messages: int = 500
    prediction_horizon: int = 1
    update_mode: Literal["append", "replace"] = "append"


@dataclass(frozen=True)
class EntityViewCfg(_ViewCfg):
    """Entity-centric view centered on one signal or message family."""

    anchor_column: str = "node_id"
    anchor_value: str | int | None = None
    history_window_size: int = 100
    future_window_size: int = 0


ViewCfg = (
    SnapshotViewCfg
    | SnapshotSequenceViewCfg
    | MultiScaleViewCfg
    | EventChunkViewCfg
    | RollingStreamViewCfg
    | EntityViewCfg
)


def view_kind(view: ViewCfg) -> str:
    """Stable human-readable label for config selection and logging."""
    if isinstance(view, SnapshotViewCfg):
        return "snapshot"
    if isinstance(view, SnapshotSequenceViewCfg):
        return "snapshot_sequence"
    if isinstance(view, MultiScaleViewCfg):
        return "multi_scale"
    if isinstance(view, EventChunkViewCfg):
        return "event_chunk"
    if isinstance(view, RollingStreamViewCfg):
        return "rolling_stream"
    if isinstance(view, EntityViewCfg):
        return "entity"
    raise TypeError(f"unsupported view config: {type(view)!r}")
