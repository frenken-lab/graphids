"""Signal profile and hypothesis-store primitives for cross-vehicle discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from .ranking import rank_signal_hypotheses, rank_signal_profiles


def _byte_cols(df: pl.DataFrame, *, prefix: str = "byte_") -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda c: int(c[len(prefix) :]) if c[len(prefix) :].isdigit() else c)


@dataclass(frozen=True)
class SignalProfileSpec:
    """How to aggregate raw CAN rows into per-signal profiles."""

    vehicle_col: str = "vehicle_id"
    signal_col: str = "arb_id"
    time_col: str = "timestamp"
    entropy_col: str = "entropy"
    attack_col: str = "attack"
    byte_prefix: str = "byte_"
    include_attack: bool = True


def build_signal_profiles(df: pl.DataFrame, spec: SignalProfileSpec | None = None) -> pl.DataFrame:
    """Aggregate raw CAN rows into one profile per vehicle/signal pair."""
    spec = spec or SignalProfileSpec()
    missing = [c for c in (spec.vehicle_col, spec.signal_col) if c not in df.columns]
    if missing:
        raise ValueError(f"build_signal_profiles missing columns: {missing}")

    sort_cols = [c for c in (spec.vehicle_col, spec.signal_col, spec.time_col) if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    byte_cols = _byte_cols(df, prefix=spec.byte_prefix)
    group_cols = [spec.vehicle_col, spec.signal_col]
    aggs: list[pl.Expr] = [pl.len().cast(pl.Int64).alias("msg_count")]
    if spec.time_col in df.columns:
        aggs.extend(
            [
                pl.col(spec.time_col).min().cast(pl.Float64).alias("timestamp_min"),
                pl.col(spec.time_col).max().cast(pl.Float64).alias("timestamp_max"),
                (pl.col(spec.time_col).max() - pl.col(spec.time_col).min())
                .cast(pl.Float64)
                .alias("duration"),
                pl.col(spec.time_col).diff().mean().cast(pl.Float64).alias("iat_mean"),
                pl.col(spec.time_col).diff().std().fill_nan(0).cast(pl.Float64).alias("iat_std"),
            ]
        )
    if spec.entropy_col in df.columns:
        aggs.extend(
            [
                pl.col(spec.entropy_col).mean().cast(pl.Float64).alias("entropy_mean"),
                pl.col(spec.entropy_col).std().fill_nan(0).cast(pl.Float64).alias("entropy_std"),
            ]
        )
    if byte_cols:
        aggs.extend(
            [
                *[pl.col(c).mean().cast(pl.Float64).alias(f"{c}_mean") for c in byte_cols],
                *[pl.col(c).std().fill_nan(0).cast(pl.Float64).alias(f"{c}_std") for c in byte_cols],
                *[pl.col(c).min().cast(pl.Float64).alias(f"{c}_min") for c in byte_cols],
                *[pl.col(c).max().cast(pl.Float64).alias(f"{c}_max") for c in byte_cols],
                *[(pl.col(c).max() - pl.col(c).min()).cast(pl.Float64).alias(f"{c}_range") for c in byte_cols],
                pl.mean_horizontal(
                    *[(pl.col(c).diff().abs().drop_nulls() > 0).mean().fill_null(0) for c in byte_cols]
                ).cast(pl.Float64).alias("change_rate"),
                pl.mean_horizontal(
                    *[pl.col(c).skew().fill_nan(0).fill_null(0).clip(-10, 10) for c in byte_cols]
                ).cast(pl.Float64).alias("skewness"),
                pl.mean_horizontal(
                    *[pl.col(c).kurtosis().fill_nan(0).fill_null(0).clip(-10, 10) for c in byte_cols]
                ).cast(pl.Float64).alias("kurtosis"),
            ]
        )
    if spec.include_attack and spec.attack_col in df.columns:
        aggs.extend(
            [
                pl.col(spec.attack_col).max().cast(pl.Int64).alias("attack_max"),
                pl.col(spec.attack_col).mean().cast(pl.Float64).alias("attack_rate"),
            ]
        )
    out = df.group_by(group_cols).agg(*aggs)
    out = out.with_columns(
        pl.concat_str(
            [pl.col(spec.vehicle_col).cast(pl.Utf8), pl.col(spec.signal_col).cast(pl.Utf8)],
            separator="::",
        ).alias("signal_key")
    )
    return out


@dataclass(frozen=True)
class SignalHypothesisSpec:
    """A provisional cross-vehicle mapping for a raw signal."""

    vehicle_id: str
    raw_signal: str
    candidate_canonical_id: str | None = None
    confidence: float = 0.0
    status: Literal["unreviewed", "provisional", "confirmed", "rejected"] = "unreviewed"
    evidence: tuple[str, ...] = ()


def initialize_hypotheses(profiles: pl.DataFrame) -> pl.DataFrame:
    """Create an empty hypothesis table aligned to a profile table."""
    required = ["vehicle_id", "arb_id", "signal_key"]
    missing = [c for c in required if c not in profiles.columns]
    if missing:
        raise ValueError(f"initialize_hypotheses missing columns: {missing}")
    return profiles.select("vehicle_id", "arb_id", "signal_key").with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("candidate_canonical_id"),
        pl.lit(0.0).cast(pl.Float64).alias("confidence"),
        pl.lit("unreviewed").alias("status"),
        pl.lit("").cast(pl.Utf8).alias("evidence"),
    )


@dataclass(frozen=True)
class DiscoveryStore:
    """File-backed signal-profile and hypothesis tables."""

    root: Path
    profiles_name: str = "signal_profiles.parquet"
    hypotheses_name: str = "canonical_hypotheses.parquet"
    manifest_name: str = "discovery_manifest.json"

    def profiles_path(self) -> Path:
        return self.root / self.profiles_name

    def hypotheses_path(self) -> Path:
        return self.root / self.hypotheses_name

    def manifest_path(self) -> Path:
        return self.root / self.manifest_name

    def write_manifest(self, *, profiles: pl.DataFrame, hypotheses: pl.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "num_profiles": int(profiles.height),
            "num_hypotheses": int(hypotheses.height),
            "profile_columns": list(profiles.columns),
            "hypothesis_columns": list(hypotheses.columns),
        }
        path = self.manifest_path()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def write_profiles(self, profiles: pl.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.profiles_path()
        profiles.write_parquet(path)
        return path

    def write_hypotheses(self, hypotheses: pl.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.hypotheses_path()
        hypotheses.write_parquet(path)
        return path

    def load_profiles(self) -> pl.DataFrame:
        return pl.read_parquet(self.profiles_path())

    def load_hypotheses(self) -> pl.DataFrame:
        return pl.read_parquet(self.hypotheses_path())

    def rank_profiles(self) -> pl.DataFrame:
        """Return ranked signal profiles from the stored profile table."""
        return rank_signal_profiles(self.load_profiles())

    def rank_hypotheses(self) -> pl.DataFrame:
        """Return ranked profiles joined to the stored hypothesis table."""
        return rank_signal_hypotheses(self.load_profiles(), self.load_hypotheses())
