"""Temporal event materialization for normalized CAN rows."""

from __future__ import annotations

import hashlib

import polars as pl
import torch
from torch_geometric.data import TemporalData

SPLIT_NAME_TO_ID: dict[str, int] = {"train": 0, "val": 1, "test": 2}
SPLIT_ID_TO_NAME: dict[int, str] = {v: k for k, v in SPLIT_NAME_TO_ID.items()}
TEMPORAL_BYTE_COLS: tuple[str, ...] = tuple(f"byte_{i}" for i in range(8))
TEMPORAL_DELTA_COLS: tuple[str, ...] = tuple(f"{c}_delta" for c in TEMPORAL_BYTE_COLS)
TEMPORAL_MSG_COL_ORDER: tuple[str, ...] = (
    *TEMPORAL_BYTE_COLS,
    *TEMPORAL_DELTA_COLS,
    "iat",
    "entropy",
    "src_is_unknown",
    "dst_is_unknown",
    "src_unknown_bucket",
    "dst_unknown_bucket",
)


def _hash_bucket(value: object, *, num_buckets: int) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets + 1


def _ensure_columns(df: pl.DataFrame, columns: dict[str, object]) -> pl.DataFrame:
    exprs = [pl.lit(default).alias(col) for col, default in columns.items() if col not in df.columns]
    return df.with_columns(exprs) if exprs else df


def _with_unknown_buckets(table: pl.DataFrame, *, num_unknown_buckets: int) -> pl.DataFrame:
    return table.with_columns(
        pl.when(pl.col("src_is_unknown"))
        .then(
            pl.col("src_raw").map_elements(
                lambda v: _hash_bucket(v, num_buckets=num_unknown_buckets),
                return_dtype=pl.Int64,
            )
        )
        .otherwise(0)
        .cast(pl.Int64)
        .alias("src_unknown_bucket"),
        pl.when(pl.col("dst_is_unknown"))
        .then(
            pl.col("dst_raw").map_elements(
                lambda v: _hash_bucket(v, num_buckets=num_unknown_buckets),
                return_dtype=pl.Int64,
            )
        )
        .otherwise(0)
        .cast(pl.Int64)
        .alias("dst_unknown_bucket"),
    )


def build_temporal_event_table(
    df: pl.DataFrame,
    *,
    id_col: str = "node_id",
    raw_id_col: str = "arb_id",
    timestamp_col: str = "timestamp",
    num_unknown_buckets: int = 128,
) -> pl.DataFrame:
    """Build one temporal event per row, preserving stream provenance.

    The first row in each stream is represented as a self-event
    (``src_id == dst_id``) so event counts match normalized row counts.
    """
    if id_col not in df.columns:
        raise ValueError(f"missing mapped id column {id_col!r}")
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column {timestamp_col!r}")

    df = _ensure_columns(
        df,
        {
            raw_id_col: "",
            "attack": 0,
            "attack_type": 0,
            "entropy": 0.0,
            "vehicle_id": "",
            "source_dir": "",
            "source_file": "",
        },
    )
    df = _ensure_columns(df, {col: 0.0 for col in TEMPORAL_BYTE_COLS})
    stream_keys = ["vehicle_id", "source_dir", "source_file"]
    rows = df.with_row_index("row_index").with_columns(
        pl.col("row_index").cast(pl.Int64),
        pl.col(id_col).cast(pl.Int64).alias("dst_id"),
        pl.col(raw_id_col).cast(pl.Utf8).alias("dst_raw"),
        pl.col(timestamp_col).cast(pl.Float64).alias("timestamp"),
        pl.col("attack").fill_null(0).cast(pl.Int64),
        pl.col("attack_type").fill_null(0).cast(pl.Int64),
        *(pl.col(c).fill_null(0).cast(pl.Float32) for c in TEMPORAL_BYTE_COLS),
        pl.col("entropy").fill_null(0).cast(pl.Float32),
    )
    streams = rows.select(stream_keys).unique(maintain_order=True).with_row_index("stream_id")
    rows = (
        rows.join(streams, on=stream_keys, how="left")
        .with_columns(pl.col("stream_id").cast(pl.Int64))
        .sort(["stream_id", "timestamp", "row_index"])
    )

    delta_exprs = [
        (pl.col(c) - pl.col(c).shift(1).over("stream_id"))
        .fill_null(0)
        .cast(pl.Float32)
        .alias(f"{c}_delta")
        for c in TEMPORAL_BYTE_COLS
    ]
    table = rows.with_columns(
        pl.col("dst_id").shift(1).over("stream_id").fill_null(pl.col("dst_id")).cast(pl.Int64).alias("src_id"),
        pl.col("dst_raw").shift(1).over("stream_id").fill_null(pl.col("dst_raw")).alias("src_raw"),
        (pl.col("timestamp") - pl.col("timestamp").shift(1).over("stream_id"))
        .fill_null(0)
        .cast(pl.Float32)
        .alias("iat"),
        *delta_exprs,
    )
    table = table.with_columns(
        (pl.col("src_id") == 0).alias("src_is_unknown"),
        (pl.col("dst_id") == 0).alias("dst_is_unknown"),
        (
            pl.col("stream_id").shift(-1).is_null()
            | (pl.col("stream_id").shift(-1) != pl.col("stream_id"))
        ).alias("reset_after"),
        (pl.col("attack") > 0).cast(pl.Int64).alias("y"),
    )
    table = _with_unknown_buckets(table, num_unknown_buckets=num_unknown_buckets)
    msg_feature_cols = [
        c
        for c in TEMPORAL_MSG_COL_ORDER
        if c
        not in {
            "src_is_unknown",
            "dst_is_unknown",
            "src_unknown_bucket",
            "dst_unknown_bucket",
        }
    ]
    return table.with_row_index("event_id").with_columns(pl.col("event_id").cast(pl.Int64)).select(
        "event_id",
        "vehicle_id",
        "source_dir",
        "source_file",
        "row_index",
        "timestamp",
        "src_id",
        "dst_id",
        "src_raw",
        "dst_raw",
        "src_is_unknown",
        "dst_is_unknown",
        "src_unknown_bucket",
        "dst_unknown_bucket",
        "stream_id",
        "reset_after",
        *msg_feature_cols,
        "y",
        "attack_type",
    )


def mark_terminal_reset(table: pl.DataFrame) -> pl.DataFrame:
    """Ensure the final event in each sliced stream resets downstream state."""
    if table.is_empty():
        return table
    if "stream_id" not in table.columns:
        last_event_id = table["event_id"][-1]
        return table.with_columns(
            pl.when(pl.col("event_id") == last_event_id)
            .then(True)
            .otherwise(pl.col("reset_after"))
            .alias("reset_after")
        )
    terminal = (
        table.group_by("stream_id")
        .agg(pl.col("event_id").max().alias("event_id"))
        .with_columns(pl.lit(True).alias("_is_terminal_event"))
    )
    return table.join(terminal, on=["stream_id", "event_id"], how="left").with_columns(
        pl.col("_is_terminal_event").fill_null(False)
    ).with_columns(
        pl.when(pl.col("_is_terminal_event"))
        .then(True)
        .otherwise(pl.col("reset_after"))
        .alias("reset_after")
    ).drop("_is_terminal_event")


def _rank_within_stream(table: pl.DataFrame) -> pl.Expr:
    return (pl.cum_count("event_id").over("stream_id") - 1).cast(pl.Int64)


def mark_split_start_self_events(table: pl.DataFrame) -> pl.DataFrame:
    """Remove transition features that cross into the start of a split."""
    if table.is_empty():
        return table
    first_in_stream = _rank_within_stream(table) == 0
    return table.with_columns(
        pl.when(first_in_stream).then(pl.col("dst_id")).otherwise(pl.col("src_id")).alias("src_id"),
        pl.when(first_in_stream).then(pl.col("dst_raw")).otherwise(pl.col("src_raw")).alias("src_raw"),
        pl.when(first_in_stream)
        .then(pl.col("dst_is_unknown"))
        .otherwise(pl.col("src_is_unknown"))
        .alias("src_is_unknown"),
        pl.when(first_in_stream)
        .then(pl.col("dst_unknown_bucket"))
        .otherwise(pl.col("src_unknown_bucket"))
        .alias("src_unknown_bucket"),
        *[
            pl.when(first_in_stream).then(0.0).otherwise(pl.col(c)).cast(pl.Float32).alias(c)
            for c in (*TEMPORAL_DELTA_COLS, "iat")
        ],
    )


def add_temporal_split_masks(
    table: pl.DataFrame,
    *,
    split_name: str,
    warmup_events: int = 0,
) -> pl.DataFrame:
    """Attach split identity plus warmup/scoring masks to an event table."""
    if split_name not in SPLIT_NAME_TO_ID:
        raise ValueError(f"unsupported split_name {split_name!r}; expected one of {sorted(SPLIT_NAME_TO_ID)}")
    if warmup_events < 0:
        raise ValueError("warmup_events must be non-negative")
    if table.is_empty():
        return table.with_columns(
            pl.lit(split_name).alias("split_name"),
            pl.lit(SPLIT_NAME_TO_ID[split_name]).cast(pl.Int64).alias("split_id"),
            pl.lit(False).alias("is_warmup"),
            pl.lit(False).alias("is_scored"),
        )
    rank = _rank_within_stream(table)
    return table.with_columns(
        pl.lit(split_name).alias("split_name"),
        pl.lit(SPLIT_NAME_TO_ID[split_name]).cast(pl.Int64).alias("split_id"),
        (rank < warmup_events).alias("is_warmup"),
    ).with_columns((~pl.col("is_warmup")).alias("is_scored"))


def _event_ids(table: pl.DataFrame) -> set[int]:
    return {int(v) for v in table["event_id"].to_list()} if "event_id" in table.columns else set()


def assert_temporal_splits_disjoint(*tables: pl.DataFrame) -> None:
    """Raise if any split shares event ids with another split."""
    seen: set[int] = set()
    for table in tables:
        ids = _event_ids(table)
        overlap = seen & ids
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(f"temporal splits share event_id values: {sample}")
        seen |= ids


def _split_event_ids_by_stream(table: pl.DataFrame, *, val_fraction: float) -> tuple[set[int], set[int]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")
    train_ids: set[int] = set()
    val_ids: set[int] = set()
    for stream in table.partition_by("stream_id", maintain_order=True):
        n_events = len(stream)
        if n_events < 2 or val_fraction == 0.0:
            train_ids.update(int(v) for v in stream["event_id"].to_list())
            continue
        n_val = max(1, int(n_events * val_fraction))
        n_val = min(n_val, n_events - 1)
        event_ids = [int(v) for v in stream["event_id"].to_list()]
        train_ids.update(event_ids[:-n_val])
        val_ids.update(event_ids[-n_val:])
    return train_ids, val_ids


def split_temporal_train_val_tables(
    table: pl.DataFrame,
    *,
    val_fraction: float,
    val_warmup_events: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Chronologically split each stream into train and validation intervals."""
    train_ids, val_ids = _split_event_ids_by_stream(table, val_fraction=val_fraction)
    train = table.filter(pl.col("event_id").is_in(train_ids))
    val = table.filter(pl.col("event_id").is_in(val_ids))
    train = add_temporal_split_masks(
        mark_terminal_reset(mark_split_start_self_events(train)),
        split_name="train",
        warmup_events=0,
    )
    val = add_temporal_split_masks(
        mark_terminal_reset(mark_split_start_self_events(val)),
        split_name="val",
        warmup_events=val_warmup_events,
    )
    assert_temporal_splits_disjoint(train, val)
    return train, val


def prepare_temporal_eval_table(
    table: pl.DataFrame,
    *,
    split_name: str = "test",
    warmup_events: int = 0,
) -> pl.DataFrame:
    """Prepare validation/test-style full streams with warmup/scoring masks."""
    return add_temporal_split_masks(
        mark_terminal_reset(mark_split_start_self_events(table)),
        split_name=split_name,
        warmup_events=warmup_events,
    )


def _tensor(df: pl.DataFrame, cols: str | list[str], *, dtype) -> torch.Tensor:
    selected = df.select(cols).fill_null(0).fill_nan(0)
    tensor = selected.to_torch(dtype=dtype)
    return tensor.squeeze(-1) if isinstance(cols, str) else tensor


def temporal_to_pyg(table: pl.DataFrame) -> TemporalData:
    """Pack a temporal event table into PyG ``TemporalData`` tensors."""
    optional_tensors = {}
    optional_specs = {
        "is_warmup": pl.Boolean,
        "is_scored": pl.Boolean,
        "split_id": pl.Int64,
    }
    for col, dtype in optional_specs.items():
        if col in table.columns:
            optional_tensors[col] = _tensor(table, col, dtype=dtype)
    data = TemporalData(
        src=_tensor(table, "src_id", dtype=pl.Int64),
        dst=_tensor(table, "dst_id", dtype=pl.Int64),
        t=_tensor(table, "timestamp", dtype=pl.Float32),
        msg=_tensor(table, list(TEMPORAL_MSG_COL_ORDER), dtype=pl.Float32),
        y=_tensor(table, "y", dtype=pl.Int64),
        attack_type=_tensor(table, "attack_type", dtype=pl.Int64),
        stream_id=_tensor(table, "stream_id", dtype=pl.Int64),
        reset_after=_tensor(table, "reset_after", dtype=pl.Boolean),
        event_id=_tensor(table, "event_id", dtype=pl.Int64),
        **optional_tensors,
    )
    if "split_name" in table.columns:
        names = table["split_name"].unique().to_list()
        data.split_name = str(names[0]) if len(names) == 1 else "mixed"
    return data
