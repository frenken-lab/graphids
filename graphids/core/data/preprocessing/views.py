"""Public view primitives for turning raw CAN/event data into graph samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._validation import require_non_negative, require_positive

ViewKind = Literal[
    "snapshot",
    "snapshot_sequence",
    "multi_scale",
    "event_chunk",
    "rolling_stream",
    "entity",
]


class _ViewCfg:
    """Marker base for public typed view configs."""


@dataclass(frozen=True)
class SnapshotViewCfg(_ViewCfg):
    """One fixed graph per sliding window."""

    window_size: int = 100
    stride: int = 100

    def __post_init__(self) -> None:
        require_positive("window_size", self.window_size)
        require_positive("stride", self.stride)


@dataclass(frozen=True)
class SnapshotSequenceViewCfg(_ViewCfg):
    """An ordered sequence of snapshot graphs."""

    window_size: int = 100
    stride: int = 100
    sequence_length: int = 4
    sequence_stride: int = 1

    def __post_init__(self) -> None:
        require_positive("window_size", self.window_size)
        require_positive("stride", self.stride)
        require_positive("sequence_length", self.sequence_length)
        require_positive("sequence_stride", self.sequence_stride)


@dataclass(frozen=True)
class MultiScaleViewCfg(_ViewCfg):
    """Parallel snapshot views at multiple window sizes."""

    window_sizes: tuple[int, ...] = (50, 100, 200)
    stride: int = 100

    def __post_init__(self) -> None:
        if not self.window_sizes:
            raise ValueError("window_sizes must not be empty")
        for window_size in self.window_sizes:
            require_positive("window_sizes", window_size)
        require_positive("stride", self.stride)


@dataclass(frozen=True)
class EventChunkViewCfg(_ViewCfg):
    """Chunk raw events by message count or duration."""

    message_count: int | None = 200
    duration_ms: float | None = None
    overlap: float = 0.0

    def __post_init__(self) -> None:
        if self.message_count is None and self.duration_ms is None:
            raise ValueError("message_count or duration_ms is required")
        if self.message_count is not None:
            require_positive("message_count", self.message_count)
        if self.duration_ms is not None:
            require_positive("duration_ms", self.duration_ms)
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0.0, 1.0)")


@dataclass(frozen=True)
class RollingStreamViewCfg(_ViewCfg):
    """Online/streaming view with bounded history."""

    history_messages: int = 500
    prediction_horizon: int = 1
    update_mode: Literal["append", "replace"] = "append"

    def __post_init__(self) -> None:
        require_positive("history_messages", self.history_messages)
        require_positive("prediction_horizon", self.prediction_horizon)


@dataclass(frozen=True)
class EntityViewCfg(_ViewCfg):
    """Entity-centric view centered on one signal or message family."""

    anchor_column: str = "node_id"
    anchor_value: str | int | None = None
    history_window_size: int = 100
    future_window_size: int = 0

    def __post_init__(self) -> None:
        if not self.anchor_column:
            raise ValueError("anchor_column must not be empty")
        require_non_negative("history_window_size", self.history_window_size)
        require_non_negative("future_window_size", self.future_window_size)
        if self.history_window_size + self.future_window_size <= 0:
            raise ValueError("history_window_size and future_window_size cannot both be zero")


ViewCfg = (
    SnapshotViewCfg
    | SnapshotSequenceViewCfg
    | MultiScaleViewCfg
    | EventChunkViewCfg
    | RollingStreamViewCfg
    | EntityViewCfg
)


def view_kind(view: ViewCfg) -> ViewKind:
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
