"""Canonical entity and feature-table primitives for cross-vehicle views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import polars as pl


def _norm(text: str) -> str:
    return text.strip().lower()


@dataclass(frozen=True)
class CanonicalEntitySpec:
    """One shared semantic entity across vehicles."""

    canonical_id: str
    name: str
    aliases: tuple[str, ...] = ()
    vehicle_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    kind: Literal["signal", "message", "state", "entity"] = "signal"
    description: str | None = None


@dataclass(frozen=True)
class CanonicalRegistry:
    """Lookup table for canonical entities and vehicle-specific aliases."""

    entities: tuple[CanonicalEntitySpec, ...]

    def __post_init__(self) -> None:
        if not self.entities:
            raise ValueError("CanonicalRegistry requires at least one entity")
        self.lookup_table()

    def lookup_table(self) -> dict[str, CanonicalEntitySpec]:
        """Return alias -> entity mapping using ``vehicle::alias`` keys."""
        table: dict[str, CanonicalEntitySpec] = {}
        for spec in self.entities:
            keys = [spec.canonical_id, spec.name, *spec.aliases]
            for key in keys:
                self._insert(table, f"*::{_norm(key)}", spec)
            for vehicle, aliases in spec.vehicle_aliases.items():
                for alias in aliases:
                    self._insert(table, f"{_norm(vehicle)}::{_norm(alias)}", spec)
        return table

    def lookup_frame(self) -> pl.DataFrame:
        """Return a canonical lookup frame for vectorized joins."""
        rows: list[dict[str, str]] = []
        for spec in self.entities:
            for key in (spec.canonical_id, spec.name, *spec.aliases):
                rows.append(
                    {
                        "vehicle_key": "*",
                        "alias_key": _norm(key),
                        "canonical_id": spec.canonical_id,
                        "canonical_name": spec.name,
                        "kind": spec.kind,
                    }
                )
            for vehicle, aliases in spec.vehicle_aliases.items():
                for alias in aliases:
                    rows.append(
                        {
                            "vehicle_key": _norm(vehicle),
                            "alias_key": _norm(alias),
                            "canonical_id": spec.canonical_id,
                            "canonical_name": spec.name,
                            "kind": spec.kind,
                        }
                    )
        return pl.DataFrame(rows).unique(["vehicle_key", "alias_key"], keep="last")

    @staticmethod
    def _insert(
        table: dict[str, CanonicalEntitySpec],
        key: str,
        spec: CanonicalEntitySpec,
    ) -> None:
        prev = table.get(key)
        if prev is not None and prev.canonical_id != spec.canonical_id:
            raise ValueError(
                f"canonical alias collision for {key!r}: {prev.canonical_id!r} != {spec.canonical_id!r}"
            )
        table[key] = spec

    def resolve(self, alias: str, *, vehicle: str | None = None) -> CanonicalEntitySpec:
        """Resolve an alias to a canonical entity."""
        table = self.lookup_table()
        alias_key = _norm(alias)
        if vehicle is not None:
            spec = table.get(f"{_norm(vehicle)}::{alias_key}")
            if spec is not None:
                return spec
        spec = table.get(f"*::{alias_key}")
        if spec is None:
            raise KeyError(f"unknown canonical alias: {alias!r}")
        return spec

    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(spec.canonical_id for spec in self.entities)

    def canonical_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.entities)


@dataclass(frozen=True)
class CanonicalFeatureFrameSpec:
    """How to flatten decoded vehicle rows into canonical feature records."""

    time_col: str = "timestamp"
    vehicle_col: str | None = "vehicle_id"
    alias_col: str = "signal"
    value_col: str = "value"
    keep_cols: tuple[str, ...] = ()
    feature_cols: tuple[str, ...] = ()
    unmapped: Literal["raise", "drop", "keep"] = "raise"


def build_canonical_feature_frame(
    df: pl.DataFrame,
    registry: CanonicalRegistry,
    *,
    spec: CanonicalFeatureFrameSpec | None = None,
) -> pl.DataFrame:
    """Normalize decoded rows into a long canonical feature table."""
    spec = spec or CanonicalFeatureFrameSpec()
    feature_cols = tuple(spec.feature_cols)
    if feature_cols:
        index_cols = [spec.time_col, *spec.keep_cols]
        if spec.vehicle_col:
            index_cols.append(spec.vehicle_col)
        index_cols = [c for i, c in enumerate(index_cols) if c in df.columns and c not in index_cols[:i]]
        frame = df.unpivot(
            on=list(feature_cols),
            index=index_cols,
            variable_name=spec.alias_col,
            value_name=spec.value_col,
        )
    else:
        frame = df.clone()
    if spec.alias_col not in frame.columns:
        raise ValueError(f"missing alias column {spec.alias_col!r}")
    if spec.value_col not in frame.columns:
        raise ValueError(f"missing value column {spec.value_col!r}")

    lookup = registry.lookup_frame()
    out = frame.with_columns(
        pl.col(spec.alias_col).cast(pl.Utf8).str.to_lowercase().alias("_alias_key"),
        pl.lit("*").alias("_vehicle_key"),
    )
    if spec.vehicle_col and spec.vehicle_col in out.columns:
        out = out.with_columns(pl.col(spec.vehicle_col).cast(pl.Utf8).str.to_lowercase().alias("_vehicle_key"))
    exact = out.join(
        lookup,
        left_on=["_vehicle_key", "_alias_key"],
        right_on=["vehicle_key", "alias_key"],
        how="left",
        suffix="_exact",
    )
    global_lookup = lookup.filter(pl.col("vehicle_key") == "*").drop("vehicle_key")
    global_join = out.join(
        global_lookup,
        left_on="_alias_key",
        right_on="alias_key",
        how="left",
        suffix="_global",
    )
    out = exact.with_columns(
        pl.coalesce(
            [pl.col("canonical_id"), global_join["canonical_id"]],
        ).alias("canonical_id"),
        pl.coalesce(
            [pl.col("canonical_name"), global_join["canonical_name"]],
        ).alias("canonical_name"),
        pl.coalesce(
            [pl.col("kind"), global_join["kind"]],
        ).alias("kind"),
    ).drop(["_alias_key", "_vehicle_key"], strict=False)

    if spec.unmapped == "raise":
        missing = out.filter(pl.col("canonical_id").is_null())
        if missing.height:
            sample = missing.select(spec.alias_col).head(5).to_series().to_list()
            raise KeyError(f"unmapped canonical aliases: {sample!r}")
    elif spec.unmapped == "drop":
        out = out.filter(pl.col("canonical_id").is_not_null())
    else:
        out = out.with_columns(
            pl.when(pl.col("canonical_id").is_null())
            .then(pl.col(spec.alias_col).cast(pl.Utf8))
            .otherwise(pl.col("canonical_id"))
            .alias("canonical_id")
        )

    spec_by_id = {e.canonical_id: e for e in registry.entities}
    out = out.with_columns(
        pl.col("canonical_id").map_elements(
            lambda cid: spec_by_id.get(str(cid)).name if cid is not None and str(cid) in spec_by_id else None,
            return_dtype=pl.Utf8,
        ).alias("canonical_name"),
        pl.col("canonical_id").map_elements(
            lambda cid: spec_by_id.get(str(cid)).kind if cid is not None and str(cid) in spec_by_id else None,
            return_dtype=pl.Utf8,
        ).alias("kind"),
    )
    return out
