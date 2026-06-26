"""Data representation configs used by preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Annotated, Literal

from pydantic import Field


def _positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class TemporalRepresentationCfg:
    kind: Literal["temporal"] = "temporal"


@dataclass(frozen=True)
class SnapshotRepresentationCfg:
    kind: Literal["snapshot"] = "snapshot"
    window_size: int = 100
    stride: int = 100

    def __post_init__(self) -> None:
        _positive("window_size", self.window_size)
        _positive("stride", self.stride)


@dataclass(frozen=True)
class SnapshotSequenceRepresentationCfg:
    kind: Literal["snapshot_sequence"] = "snapshot_sequence"
    window_size: int = 100
    stride: int = 100
    sequence_length: int = 4
    sequence_stride: int = 1

    def __post_init__(self) -> None:
        _positive("window_size", self.window_size)
        _positive("stride", self.stride)
        _positive("sequence_length", self.sequence_length)
        _positive("sequence_stride", self.sequence_stride)


GraphRepresentationCfg = Annotated[
    TemporalRepresentationCfg | SnapshotRepresentationCfg | SnapshotSequenceRepresentationCfg,
    Field(discriminator="kind"),
]


def representation_kind(cfg: GraphRepresentationCfg) -> str:
    return cfg.kind


def representation_payload(cfg: GraphRepresentationCfg) -> dict[str, object]:
    return asdict(cfg)


def representation_digest(cfg: GraphRepresentationCfg) -> str:
    payload = json.dumps(representation_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def representation_window_defaults(cfg: GraphRepresentationCfg) -> tuple[int, int]:
    if isinstance(cfg, TemporalRepresentationCfg):
        raise ValueError("temporal representation does not have window_size/stride")
    return cfg.window_size, cfg.stride
