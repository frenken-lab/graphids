"""Signal profile artifacts written beside graph caches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


def _byte_cols(df: pl.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("byte_")]
    return sorted(cols, key=lambda c: int(c[5:]) if c[5:].isdigit() else c)


def build_signal_profiles(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw CAN rows into one profile per vehicle/arbitration ID."""

    missing = [c for c in ("vehicle_id", "arb_id") if c not in df.columns]
    if missing:
        raise ValueError(f"build_signal_profiles missing columns: {missing}")

    sort_cols = [c for c in ("vehicle_id", "arb_id", "timestamp") if c in df.columns]
    if sort_cols:
        df = df.sort(sort_cols)

    aggs: list[pl.Expr] = [pl.len().cast(pl.Int64).alias("msg_count")]
    if "timestamp" in df.columns:
        aggs.extend(
            [
                pl.col("timestamp").min().cast(pl.Float64).alias("timestamp_min"),
                pl.col("timestamp").max().cast(pl.Float64).alias("timestamp_max"),
                (pl.col("timestamp").max() - pl.col("timestamp").min()).cast(pl.Float64).alias("duration"),
                pl.col("timestamp").diff().mean().cast(pl.Float64).alias("iat_mean"),
                pl.col("timestamp").diff().std().fill_nan(0).cast(pl.Float64).alias("iat_std"),
            ]
        )
    if "entropy" in df.columns:
        aggs.extend(
            [
                pl.col("entropy").mean().cast(pl.Float64).alias("entropy_mean"),
                pl.col("entropy").std().fill_nan(0).cast(pl.Float64).alias("entropy_std"),
            ]
        )

    byte_cols = _byte_cols(df)
    aggs.extend(pl.col(c).mean().cast(pl.Float64).alias(f"{c}_mean") for c in byte_cols)
    aggs.extend(pl.col(c).std().fill_nan(0).cast(pl.Float64).alias(f"{c}_std") for c in byte_cols)
    aggs.extend((pl.col(c).max() - pl.col(c).min()).cast(pl.Float64).alias(f"{c}_range") for c in byte_cols)
    if byte_cols:
        aggs.append(
            pl.mean_horizontal(
                *[(pl.col(c).diff().abs().drop_nulls() > 0).mean().fill_null(0) for c in byte_cols]
            ).cast(pl.Float64).alias("change_rate")
        )
    if "attack" in df.columns:
        aggs.extend(
            [
                pl.col("attack").max().cast(pl.Int64).alias("attack_max"),
                pl.col("attack").mean().cast(pl.Float64).alias("attack_rate"),
            ]
        )

    return df.group_by("vehicle_id", "arb_id").agg(*aggs).with_columns(
        pl.concat_str([pl.col("vehicle_id").cast(pl.Utf8), pl.col("arb_id").cast(pl.Utf8)], separator="::").alias("signal_key")
    )


def initialize_hypotheses(profiles: pl.DataFrame) -> pl.DataFrame:
    """Create empty editable mapping rows for profile review."""

    required = ["vehicle_id", "arb_id", "signal_key"]
    missing = [c for c in required if c not in profiles.columns]
    if missing:
        raise ValueError(f"initialize_hypotheses missing columns: {missing}")
    return profiles.select(*required).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("candidate_canonical_id"),
        pl.lit(0.0).cast(pl.Float64).alias("confidence"),
        pl.lit("unreviewed").alias("status"),
        pl.lit("").cast(pl.Utf8).alias("evidence"),
    )


@dataclass(frozen=True)
class DiscoveryStore:
    root: Path

    def profiles_path(self) -> Path:
        return self.root / "signal_profiles.parquet"

    def hypotheses_path(self) -> Path:
        return self.root / "canonical_hypotheses.parquet"

    def write_profiles(self, profiles: pl.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        profiles.write_parquet(self.profiles_path())
        return self.profiles_path()

    def write_hypotheses(self, hypotheses: pl.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        hypotheses.write_parquet(self.hypotheses_path())
        return self.hypotheses_path()

    def load_profiles(self) -> pl.DataFrame:
        return pl.read_parquet(self.profiles_path())

    def load_hypotheses(self) -> pl.DataFrame:
        return pl.read_parquet(self.hypotheses_path())
