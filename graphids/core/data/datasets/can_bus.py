"""CAN bus dataset — feature schema, payload parser, source.

Everything CAN-bus-specific lives here: hex payload parsing, byte-column
feature expressions, attack-type taxonomy, and the ``CANBusDataset`` /
``CANBusSource`` adapters. Generic graph-dataset orchestration (splits,
scaler fit/apply, metadata merge, mmap load) lives in ``_base.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from structlog import get_logger

from graphids.core.data.datasets._base import (
    BaseGraphDataset,
    BaseGraphSource,
    GraphSchema,
)

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Attack-type taxonomy
# ---------------------------------------------------------------------------

# Insertion order matters: ``_infer_attack_type`` does substring matching and
# returns the first hit, so longer/more-specific keys must precede their
# shorter prefixes (``rpm-accessory`` before ``rpm``, ``speed-accessory``
# before ``speed``). ``force-neutral`` aliases the ``gear`` family under the
# can-train-and-test v1.5 filename convention.
ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0,
    "attack_free": 0,
    "benign": 0,
    "dos": 1,
    "fuzzy": 2,
    "fuzzing": 2,
    "force-neutral": 3,
    "gear": 3,
    "rpm-accessory": 12,
    "rpm": 4,
    "flooding": 5,
    "malfunction": 6,
    "double": 7,
    "triple": 8,
    "interval": 9,
    "speed-accessory": 11,
    "speed": 10,
    "standstill": 13,
    "systematic": 14,
    "suppress": 15,
    "masquerade": 16,
}

ATTACK_TYPE_NAMES: dict[int, str] = {v: k for k, v in ATTACK_TYPE_CODES.items() if v != 0}
ATTACK_TYPE_NAMES[0] = "benign"


# ---------------------------------------------------------------------------
# CAN feature schema — column layouts, Polars expressions, helper fns
# ---------------------------------------------------------------------------

BYTE_COLS = [f"byte_{i}" for i in range(8)]

# Column order defines tensor layout. Changing order changes model input.
NODE_COL_ORDER = (
    [f"{c}_mean" for c in BYTE_COLS]
    + [f"{c}_std" for c in BYTE_COLS]
    + [f"{c}_range" for c in BYTE_COLS]
    + [
        "msg_count",
        "entropy_mean",
        "skewness",
        "kurtosis",
        "clustering_coeff",
        "split_half_ratio",
        "change_rate",
        "node_iat_mean",
        "node_iat_std",
        "in_degree",
        "out_degree",
    ]
)

N_NODE_FEATURES = len(NODE_COL_ORDER)
# Edge feature layout: iat + 8 byte diffs + bidirectional flag + freq.
EDGE_COL_ORDER = (
    "iat",
    *(f"byte_{i}_diff" for i in range(8)),
    "bidir",
    "edge_freq",
)

N_EDGE_FEATURES = len(EDGE_COL_ORDER)  # 11

# Polars aggregation expressions for per-node stats within a window.
# Used by group_by("node_id").agg() and group_by(["_wid", "node_id"]).agg().
# Requires columns: byte_0..7, entropy, _first_half (bool).
NODE_STAT_EXPRS: list[pl.Expr] = [
    *[pl.col(c).mean().alias(f"{c}_mean") for c in BYTE_COLS],
    *[pl.col(c).std().alias(f"{c}_std") for c in BYTE_COLS],
    *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in BYTE_COLS],
    pl.len().cast(pl.Float32).alias("msg_count"),
    pl.col("entropy").mean().alias("entropy_mean"),
    pl.mean_horizontal(*[pl.col(c).skew().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]).alias(
        "skewness"
    ),
    pl.mean_horizontal(*[pl.col(c).kurtosis().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]).alias(
        "kurtosis"
    ),
    pl.lit(0.0).alias("clustering_coeff"),  # filled per-window from graph structure
    pl.col("_first_half").mean().alias("split_half_ratio"),
    pl.mean_horizontal(
        *[(pl.col(c).diff().abs().drop_nulls() > 0).mean() for c in BYTE_COLS]
    ).alias("change_rate"),
    pl.col("timestamp").diff().mean().cast(pl.Float32).alias("node_iat_mean"),
    pl.col("timestamp").diff().std().fill_nan(0).cast(pl.Float32).alias("node_iat_std"),
    pl.lit(0.0).alias("in_degree"),  # filled post-hoc from edge_index
    pl.lit(0.0).alias("out_degree"),  # filled post-hoc from edge_index
]

# Polars expressions for vectorized edge feature computation.
# Used by with_columns() after sort(["_wid", "_row"]).
# Requires columns: timestamp, byte_0..7, _wid.
# Note: bidir is computed separately via self-join (not expressible as a single expression).
EDGE_STAT_EXPRS: list[pl.Expr] = [
    pl.col("timestamp").diff().cast(pl.Float32).alias("iat"),
    *[pl.col(f"byte_{i}").diff().abs().cast(pl.Float32).alias(f"byte_{i}_diff") for i in range(8)],
]

# Label aggregations per window: y (binary attack) + attack_type (multiclass).
LABEL_EXPRS: list[pl.Expr] = [
    (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
    pl.col("attack_type")
    .filter(pl.col("attack_type") > 0)
    .mode()
    .first()
    .fill_null(0)
    .alias("attack_type"),
]

# Columns required by edge-feature computation (byte diffs need byte_0..7).
EDGE_BASE_COLS: list[str] = [f"byte_{i}" for i in range(8)]


CAN_SCHEMA = GraphSchema(
    node_stat_exprs=NODE_STAT_EXPRS,
    edge_stat_exprs=EDGE_STAT_EXPRS,
    node_col_order=NODE_COL_ORDER,
    edge_col_order=EDGE_COL_ORDER,
    label_exprs=LABEL_EXPRS,
    edge_base_cols=EDGE_BASE_COLS,
    vocab_column="arb_id",
    attack_type_codes=ATTACK_TYPE_CODES,
    attack_type_names=ATTACK_TYPE_NAMES,
)


def parse_payload(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Parse hex payload column into 8 byte columns + Shannon entropy.

    Expects a 'payload' column (16-char hex string). Adds byte_0..byte_7
    (Float32) and entropy (Float32). Passthrough if byte_0 already exists.
    """
    if "byte_0" in lf.collect_schema().names():
        return lf
    byte_exprs = [
        pl.col("payload")
        .str.slice(i * 2, 2)
        .str.to_integer(base=16, strict=False)
        .fill_null(0)
        .cast(pl.Float32)
        .alias(f"byte_{i}")
        for i in range(8)
    ]
    lf = lf.with_columns(byte_exprs)
    byte_cols = [pl.col(f"byte_{i}") for i in range(8)]
    row_sum = pl.sum_horizontal(byte_cols).clip(1e-12, None)
    entropy_terms = [
        pl.when(c > 0).then(-(c / row_sum) * (c / row_sum).log()).otherwise(0.0) for c in byte_cols
    ]
    return lf.with_columns(pl.sum_horizontal(entropy_terms).alias("entropy"))


# ---------------------------------------------------------------------------
# CANBusDataset — only CAN-specific override is _read_raw
# ---------------------------------------------------------------------------


class CANBusDataset(BaseGraphDataset):
    """CAN bus intrusion detection graph dataset.

    Each graph is one sliding window of CAN messages. Nodes are arbitration
    IDs, edges are temporal adjacency (shift-1).
    """

    SCHEMA: ClassVar[GraphSchema] = CAN_SCHEMA

    def _read_raw(self) -> pl.DataFrame:
        """Lazy-scan CSVs from declared source_dirs, parse hex, tag attack types.

        Scope is explicit: only subdirs in ``self.source_dirs`` are read.
        Recursive glob over ``raw_data_dir`` would silently pull every
        train+test CSV into one tensor (contamination).
        """
        if not self.source_dirs:
            raise ValueError(
                f"CANBusDataset split={self.split!r} has no source_dirs; "
                "cannot build cache from raw CSVs. Caller must pass "
                "source_dirs=[...] at construction."
            )
        frames = []
        for sub in self.source_dirs:
            sub_path = self.raw_data_dir / sub
            if not sub_path.is_dir():
                raise FileNotFoundError(
                    f"Declared source_dir {sub!r} missing under {self.raw_data_dir}"
                )
            for csv_path in sorted(sub_path.rglob("*.csv")):
                at = self._infer_attack_type(csv_path)
                lf = pl.scan_csv(csv_path).with_columns(pl.lit(at).alias("attack_type"))
                frames.append(lf)
        if not frames:
            raise ValueError(f"No CSVs under any of {self.source_dirs!r} in {self.raw_data_dir}")

        combined = pl.concat(frames).sort("timestamp")

        # Normalize column names: HCRL CSVs use different names than our schema
        col_names = combined.collect_schema().names()
        renames = {}
        if "arbitration_id" in col_names:
            renames["arbitration_id"] = "arb_id"
        if "data_field" in col_names:
            renames["data_field"] = "payload"
        if renames:
            combined = combined.rename(renames)

        combined = parse_payload(combined)

        return combined.collect()


# ---------------------------------------------------------------------------
# CANBusSource — KIND + DATASET_CLS + scan_arb_ids hook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CANBusSource(BaseGraphSource):
    """CAN bus dataset source — produces train/val/test splits on demand.

    ``get_or_build`` in ``graphids.core.data.state`` memoizes the
    ``DatasetState`` returned by ``build()`` under ``cache_key`` so
    multi-stage runs sharing a process pay preprocessing + mmap cost
    once instead of per-stage.

    ``name`` is a catalog entry (e.g. ``hcrl_sa``, ``set_01``). The
    catalog is loaded at build time via
    ``graphids.config.catalog.load_catalog`` — no name validation at
    construction, since the catalog may shift.
    """

    KIND: ClassVar[str] = "canbus"
    DATASET_CLS: ClassVar[type[BaseGraphDataset]] = CANBusDataset

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        # Tolerates both the HCRL ``arbitration_id`` and the in-schema
        # ``arb_id`` column names — the file-level scanner handles the
        # alias.
        from graphids.core.data.preprocessing.vocab import scan_arb_ids

        return scan_arb_ids(raw_dir, source_dirs)
