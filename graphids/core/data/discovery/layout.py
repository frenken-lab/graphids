"""Storage-layout primitives for raw events, views, and hypotheses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

ViewKind = Literal["snapshot", "snapshot_sequence"]


@dataclass(frozen=True)
class RawEventTableSpec:
    """Canonical storage for immutable decoded CAN rows."""

    root: Path
    name: str = "raw_can_events"
    partition_cols: tuple[str, ...] = ("vehicle_id", "day")
    format: Literal["parquet", "sqlite", "duckdb"] = "parquet"
    primary_time_col: str = "timestamp"
    raw_id_col: str = "arb_id"
    vehicle_col: str = "vehicle_id"
    attack_col: str = "attack"
    signal_hint_col: str = "signal_hint"

    def path(self) -> Path:
        return self.root / self.name


@dataclass(frozen=True)
class MaterializedViewSpec:
    """Training-facing view materialization contract."""

    root: Path
    name: str = "materialized_views"
    view_kind: ViewKind = "snapshot"
    partition_cols: tuple[str, ...] = ("vehicle_id", "view_kind", "split")
    format: Literal["parquet", "sqlite", "duckdb"] = "parquet"
    key_cols: tuple[str, ...] = ("vehicle_id", "canonical_id", "timestamp")
    feature_cols: tuple[str, ...] = ()
    label_cols: tuple[str, ...] = ("attack", "attack_type")

    def path(self) -> Path:
        return self.root / self.name / self.view_kind


@dataclass(frozen=True)
class HypothesisRecordSpec:
    """Provisional semantic mapping for a raw signal."""

    vehicle_id: str
    raw_signal: str
    candidate_canonical_id: str | None = None
    confidence: float = 0.0
    status: Literal["unreviewed", "provisional", "confirmed", "rejected"] = "unreviewed"
    evidence: tuple[str, ...] = ()
    profile_path: Path | None = None
    feature_digest: str | None = None


@dataclass(frozen=True)
class DataLayerLayout:
    """A single place to point at the three persistent data layers."""

    root: Path
    raw: RawEventTableSpec = field(init=False)
    views: MaterializedViewSpec = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw", RawEventTableSpec(root=self.root))
        object.__setattr__(self, "views", MaterializedViewSpec(root=self.root))

    @property
    def hypotheses_path(self) -> Path:
        return self.root / "discovery" / "canonical_hypotheses.parquet"

    @property
    def profiles_path(self) -> Path:
        return self.root / "discovery" / "signal_profiles.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.root / "discovery" / "discovery_manifest.json"
