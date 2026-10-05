"""Strict physical-time event matching shared by relationship miners."""
from __future__ import annotations

import polars as pl


def preceding_lags(targets: list[float], triggers: list[float], horizon: float) -> list[float]:
    """Lag from each target to its latest strictly earlier trigger in range."""
    if not targets or not triggers:
        return []
    left = pl.DataFrame({"target": pl.Series(targets, dtype=pl.Float64)})
    right = pl.DataFrame({"trigger": pl.Series(triggers, dtype=pl.Float64)})
    joined = left.join_asof(right, left_on="target", right_on="trigger",
                            strategy="backward", allow_exact_matches=False,
                            tolerance=horizon)
    return (joined.select((pl.col("target") - pl.col("trigger")).alias("lag"))
            .drop_nulls()["lag"].to_list())


def following_lags(starts: list[float], ends: list[float], low: float, high: float) -> list[float]:
    """Lag to each start's first end strictly after start+low, up to high."""
    if not starts or not ends:
        return []
    left = pl.DataFrame({"start": pl.Series(starts, dtype=pl.Float64)}).with_columns(
        (pl.col("start") + low).alias("lower"))
    right = pl.DataFrame({"end": pl.Series(ends, dtype=pl.Float64)})
    joined = left.join_asof(right, left_on="lower", right_on="end", strategy="forward",
                            allow_exact_matches=False, tolerance=high - low)
    return (joined.select((pl.col("end") - pl.col("start")).alias("lag"))
            .drop_nulls()["lag"].to_list())
