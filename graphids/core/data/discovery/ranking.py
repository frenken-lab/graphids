"""Ranking helpers for cross-vehicle signal discovery."""

from __future__ import annotations

import polars as pl


def _pick_group_col(df: pl.DataFrame) -> str:
    for candidate in ("signal_key", "arb_id", "raw_signal", "canonical_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "rank_signal_profiles requires one of signal_key, arb_id, raw_signal, or canonical_id"
    )


def rank_signal_profiles(profiles: pl.DataFrame) -> pl.DataFrame:
    """Score per-signal profile rows by cross-vehicle support and stability.

    The goal is not to claim a final ontology match; it is to give the
    discovery layer a concrete relational ranking pass that can surface
    stable signals for review.
    """
    group_col = _pick_group_col(profiles)
    required = {group_col, "vehicle_id"}
    missing = sorted(required.difference(profiles.columns))
    if missing:
        raise ValueError(f"rank_signal_profiles missing columns: {missing}")

    optional_defaults = {
        "msg_count": 0.0,
        "entropy_mean": 0.0,
        "entropy_std": 0.0,
        "change_rate": 0.0,
        "attack_rate": 0.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
    }
    for col, default in optional_defaults.items():
        if col not in profiles.columns:
            profiles = profiles.with_columns(pl.lit(default).cast(pl.Float64).alias(col))

    support = (
        profiles.group_by(group_col)
        .agg(
            pl.n_unique("vehicle_id").alias("vehicle_support"),
            pl.len().alias("profile_rows"),
            pl.col("msg_count").mean().fill_null(0).cast(pl.Float64).alias("msg_count_mean"),
            pl.col("msg_count").std().fill_null(0).cast(pl.Float64).alias("msg_count_std"),
            pl.col("entropy_mean").mean().fill_null(0).cast(pl.Float64).alias("entropy_mean"),
            pl.col("entropy_std").mean().fill_null(0).cast(pl.Float64).alias("entropy_std"),
            pl.col("change_rate").mean().fill_null(0).cast(pl.Float64).alias("change_rate"),
            pl.col("attack_rate").mean().fill_null(0).cast(pl.Float64).alias("attack_rate"),
            pl.col("skewness").mean().fill_null(0).cast(pl.Float64).alias("skewness"),
            pl.col("kurtosis").mean().fill_null(0).cast(pl.Float64).alias("kurtosis"),
        )
        .with_columns(
            (
                pl.col("vehicle_support").cast(pl.Float64)
                * (1.0 + (pl.col("msg_count_mean") + 1.0).log())
                / (1.0 + pl.col("msg_count_std"))
                / (1.0 + pl.col("entropy_std"))
                / (1.0 + pl.col("change_rate").abs())
                / (1.0 + pl.col("attack_rate").clip(0.0, 1.0))
            ).alias("ranking_score")
        )
        .sort(["ranking_score", "vehicle_support", "profile_rows"], descending=True)
    )
    return support


def rank_signal_hypotheses(
    profiles: pl.DataFrame,
    hypotheses: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join profile scores to provisional hypotheses when available."""
    ranked = rank_signal_profiles(profiles)
    if hypotheses is None or hypotheses.is_empty():
        return ranked

    join_col = _pick_group_col(profiles)
    hyp_cols = [c for c in ("vehicle_id", "raw_signal", "candidate_canonical_id", "status", "confidence") if c in hypotheses.columns]
    if join_col in hypotheses.columns:
        cols = [join_col, *[c for c in hyp_cols if c != join_col]]
        return ranked.join(hypotheses.select(*cols), on=join_col, how="left")
    if "raw_signal" in hypotheses.columns and join_col != "raw_signal":
        cols = ["raw_signal", *[c for c in hyp_cols if c != "raw_signal"]]
        return ranked.join(hypotheses.select(*cols), left_on=join_col, right_on="raw_signal", how="left")
    return ranked
