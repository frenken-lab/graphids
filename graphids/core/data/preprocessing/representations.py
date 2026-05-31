"""Explicit graph-representation configs for training and discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from pydantic import Field

from .segments import (
    EntitySegmentCfg,
    MultiScaleSegmentCfg,
    SequenceSegmentCfg,
    WindowSegmentCfg,
)
from .temporal import TemporalGraphSpec
from .views import (
    EntityViewCfg,
    MultiScaleViewCfg,
    RollingStreamViewCfg,
    SnapshotSequenceViewCfg,
    SnapshotViewCfg,
    ViewCfg,
)


@dataclass(frozen=True)
class SnapshotRepresentationCfg:
    """One graph per sliding window."""

    kind: Literal["snapshot"] = "snapshot"
    window_size: int = 100
    stride: int = 100


@dataclass(frozen=True)
class SnapshotSequenceRepresentationCfg:
    """Ordered sequence of snapshot graphs."""

    kind: Literal["snapshot_sequence"] = "snapshot_sequence"
    window_size: int = 100
    stride: int = 100
    sequence_length: int = 4
    sequence_stride: int = 1


@dataclass(frozen=True)
class MultiScaleRepresentationCfg:
    """Parallel snapshots at multiple window sizes."""

    kind: Literal["multi_scale"] = "multi_scale"
    window_sizes: tuple[int, ...] = (50, 100, 200)
    stride: int = 100


@dataclass(frozen=True)
class TemporalRepresentationCfg:
    """Event stream representation built as PyG ``TemporalData``."""

    kind: Literal["temporal"] = "temporal"
    time_col: str = "timestamp"
    binary_target: bool = True
    history_messages: int | None = None


@dataclass(frozen=True)
class EntityRepresentationCfg:
    """Entity-centric representation centered on one signal or message family."""

    kind: Literal["entity"] = "entity"
    anchor_column: str = "node_id"
    anchor_value: str | int | None = None
    history_window_size: int = 100
    future_window_size: int = 0


GraphRepresentationCfg = Annotated[
    SnapshotRepresentationCfg
    | SnapshotSequenceRepresentationCfg
    | MultiScaleRepresentationCfg
    | TemporalRepresentationCfg
    | EntityRepresentationCfg,
    Field(discriminator="kind"),
]


@dataclass(frozen=True)
class GraphRepresentationPlan:
    """Resolved representation kind and config payload."""

    kind: Literal["snapshot", "snapshot_sequence", "multi_scale", "temporal", "entity"]
    cfg: GraphRepresentationCfg


def representation_kind(cfg: GraphRepresentationCfg) -> str:
    """Stable label for logging and config routing."""
    if isinstance(cfg, SnapshotRepresentationCfg):
        return "snapshot"
    if isinstance(cfg, SnapshotSequenceRepresentationCfg):
        return "snapshot_sequence"
    if isinstance(cfg, MultiScaleRepresentationCfg):
        return "multi_scale"
    if isinstance(cfg, TemporalRepresentationCfg):
        return "temporal"
    if isinstance(cfg, EntityRepresentationCfg):
        return "entity"
    raise TypeError(f"unsupported representation config: {type(cfg)!r}")


def representation_payload(cfg: GraphRepresentationCfg) -> dict[str, object]:
    """Stable JSON-serializable payload for cache identity and metadata."""
    return asdict(cfg)


def representation_digest(cfg: GraphRepresentationCfg) -> str:
    """Short stable digest for paths and cache keys."""
    payload = json.dumps(representation_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def representation_window_defaults(cfg: GraphRepresentationCfg) -> tuple[int, int]:
    """Derive legacy window knobs from the explicit representation config."""
    if isinstance(cfg, SnapshotRepresentationCfg):
        return cfg.window_size, cfg.stride
    if isinstance(cfg, SnapshotSequenceRepresentationCfg):
        return cfg.window_size, cfg.stride
    if isinstance(cfg, MultiScaleRepresentationCfg):
        return min(cfg.window_sizes), cfg.stride
    if isinstance(cfg, EntityRepresentationCfg):
        return (
            cfg.history_window_size + cfg.future_window_size + 1,
            max(1, cfg.future_window_size or 1),
        )
    if isinstance(cfg, TemporalRepresentationCfg):
        return 100, 100
    raise TypeError(f"unsupported representation config: {type(cfg)!r}")


def representation_plan(cfg: GraphRepresentationCfg) -> GraphRepresentationPlan:
    """Wrap a representation config with its stable kind label."""
    return GraphRepresentationPlan(kind=representation_kind(cfg), cfg=cfg)


def representation_view(cfg: GraphRepresentationCfg) -> ViewCfg:
    """Map a representation config to the corresponding public view config."""
    if isinstance(cfg, SnapshotRepresentationCfg):
        return SnapshotViewCfg(window_size=cfg.window_size, stride=cfg.stride)
    if isinstance(cfg, SnapshotSequenceRepresentationCfg):
        return SnapshotSequenceViewCfg(
            window_size=cfg.window_size,
            stride=cfg.stride,
            sequence_length=cfg.sequence_length,
            sequence_stride=cfg.sequence_stride,
        )
    if isinstance(cfg, MultiScaleRepresentationCfg):
        return MultiScaleViewCfg(window_sizes=cfg.window_sizes, stride=cfg.stride)
    if isinstance(cfg, TemporalRepresentationCfg):
        return RollingStreamViewCfg(
            history_messages=cfg.history_messages or 500,
            prediction_horizon=1,
        )
    if isinstance(cfg, EntityRepresentationCfg):
        return EntityViewCfg(
            anchor_column=cfg.anchor_column,
            anchor_value=cfg.anchor_value,
            history_window_size=cfg.history_window_size,
            future_window_size=cfg.future_window_size,
        )
    raise TypeError(f"unsupported representation config: {type(cfg)!r}")


def representation_segment(
    cfg: GraphRepresentationCfg,
) -> WindowSegmentCfg | SequenceSegmentCfg | MultiScaleSegmentCfg | EntitySegmentCfg:
    """Map a representation config to the corresponding segment primitive."""
    if isinstance(cfg, SnapshotRepresentationCfg):
        return WindowSegmentCfg(window_size=cfg.window_size, stride=cfg.stride)
    if isinstance(cfg, SnapshotSequenceRepresentationCfg):
        return SequenceSegmentCfg(
            window_size=cfg.window_size,
            stride=cfg.stride,
            sequence_length=cfg.sequence_length,
            sequence_stride=cfg.sequence_stride,
        )
    if isinstance(cfg, MultiScaleRepresentationCfg):
        return MultiScaleSegmentCfg(window_sizes=cfg.window_sizes, stride=cfg.stride)
    if isinstance(cfg, EntityRepresentationCfg):
        return EntitySegmentCfg(
            anchor_column=cfg.anchor_column,
            anchor_value=cfg.anchor_value,
            history_window_size=cfg.history_window_size,
            future_window_size=cfg.future_window_size,
        )
    raise TypeError(
        f"representation {type(cfg).__name__} does not map to a segment primitive"
    )


def representation_temporal_spec(cfg: GraphRepresentationCfg) -> TemporalGraphSpec:
    """Map a representation config to the temporal-stream spec."""
    if isinstance(cfg, TemporalRepresentationCfg):
        return TemporalGraphSpec(
            time_col=cfg.time_col,
            binary_target=cfg.binary_target,
            feature_cols=(),
            target_col="attack",
            aux_label_cols=("attack_type",),
        )
    raise TypeError(
        f"representation {type(cfg).__name__} does not map to a temporal spec"
    )
