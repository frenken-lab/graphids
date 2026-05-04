"""CAN bus dataset — protocol parser + domain features + adapter classes.

Module is sectioned so a reader knows what's CAN-specific vs what's
graph-pipeline plumbing that just happens to live here:

  §1. CAN protocol constants    — byte layout (CAN-specific)
  §2. Attack taxonomy           — code/name maps (CAN-specific)
  §3. Domain feature expressions — Polars aggs over CAN data (CAN-specific)
  §4. Topology placeholders     — filled by GraphPipeline (NOT CAN-specific)
  §5. Tensor column orders      — final layouts (concat of §3 + §4)
  §6. Payload parser            — hex → byte_0..7 + entropy (CAN-specific)
  §7. Adapter classes           — _read_raw + _scan_vocab hooks

If GraphPipeline ever auto-injects §4 (clustering_coeff, in/out degree)
via a generic ``WithGraphTopology`` schema mixin, §4 disappears from
this file and adapters only declare domain knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
from structlog import get_logger

from graphids.core.data.datasets._base import BaseGraphDataset, BaseGraphSource, GraphSchema

log = get_logger(__name__)


# ─── §1. CAN protocol constants ───────────────────────────────────────

N_BYTES = 8
BYTE_COLS = [f"byte_{i}" for i in range(N_BYTES)]


# ─── §2. Attack taxonomy ──────────────────────────────────────────────

# Insertion order matters: ``_infer_attack_type`` does substring match and
# returns the first hit, so longer/more-specific keys precede their prefixes.
ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0, "attack_free": 0, "benign": 0,
    "dos": 1,
    "fuzzy": 2, "fuzzing": 2,
    "force-neutral": 3, "gear": 3,
    "rpm-accessory": 12,
    "rpm": 4,
    "flooding": 5,
    "malfunction": 6,
    "double": 7, "triple": 8, "interval": 9,
    "speed-accessory": 11,
    "speed": 10,
    "standstill": 13,
    "systematic": 14,
    "suppress": 15,
    "masquerade": 16,
}
ATTACK_TYPE_NAMES: dict[int, str] = {v: k for k, v in ATTACK_TYPE_CODES.items() if v != 0}
ATTACK_TYPE_NAMES[0] = "benign"


# ─── §3. Domain feature expressions (CAN-specific) ────────────────────

# Per-node aggs within a window. Inputs: byte_0..7, entropy, _first_half, timestamp.
DOMAIN_NODE_EXPRS: list[pl.Expr] = [
    *[pl.col(c).mean().alias(f"{c}_mean") for c in BYTE_COLS],
    *[pl.col(c).std().alias(f"{c}_std") for c in BYTE_COLS],
    *[(pl.col(c).max() - pl.col(c).min()).alias(f"{c}_range") for c in BYTE_COLS],
    pl.len().cast(pl.Float32).alias("msg_count"),
    pl.col("entropy").mean().alias("entropy_mean"),
    # Skew/kurtosis clamped — fp16 max ~65504, raw values can hit 1e17.
    pl.mean_horizontal(
        *[pl.col(c).skew().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]
    ).alias("skewness"),
    pl.mean_horizontal(
        *[pl.col(c).kurtosis().fill_nan(0).clip(-10, 10) for c in BYTE_COLS]
    ).alias("kurtosis"),
    pl.col("_first_half").mean().alias("split_half_ratio"),
    pl.mean_horizontal(
        *[(pl.col(c).diff().abs().drop_nulls() > 0).mean() for c in BYTE_COLS]
    ).alias("change_rate"),
    pl.col("timestamp").diff().mean().cast(pl.Float32).alias("node_iat_mean"),
    pl.col("timestamp").diff().std().fill_nan(0).cast(pl.Float32).alias("node_iat_std"),
]

# Per-edge feature exprs (within window, after sort). Inputs: timestamp, byte_0..7.
DOMAIN_EDGE_EXPRS: list[pl.Expr] = [
    pl.col("timestamp").diff().cast(pl.Float32).alias("iat"),
    *[pl.col(c).diff().abs().cast(pl.Float32).alias(f"{c}_diff") for c in BYTE_COLS],
]

# Per-window labels: y (binary attack) + attack_type (multiclass mode).
LABEL_EXPRS: list[pl.Expr] = [
    (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
    pl.col("attack_type")
    .filter(pl.col("attack_type") > 0)
    .mode()
    .first()
    .fill_null(0)
    .alias("attack_type"),
]


# ─── §4. Topology placeholders (filled by GraphPipeline, not CAN) ─────

# These columns hold ``0.0`` until GraphPipeline._compute_graph_structure
# overwrites them via ``DataFrame.update``. They live here only because
# the schema must declare every column the tensor select reads from.
# Move to a ``WithGraphTopology`` schema mixin to delete this section.
TOPOLOGY_NODE_PLACEHOLDERS: list[pl.Expr] = [
    pl.lit(0.0).alias("clustering_coeff"),
    pl.lit(0.0).alias("in_degree"),
    pl.lit(0.0).alias("out_degree"),
]
# bidir + edge_freq are filled inside the pipeline's edge aggregation
# (no placeholder needed — they're added as real columns there).


# ─── §5. Tensor column orders ─────────────────────────────────────────

NODE_COL_ORDER = (
    [f"{c}_mean" for c in BYTE_COLS]
    + [f"{c}_std" for c in BYTE_COLS]
    + [f"{c}_range" for c in BYTE_COLS]
    + ["msg_count", "entropy_mean", "skewness", "kurtosis",
       "clustering_coeff",  # topology
       "split_half_ratio", "change_rate", "node_iat_mean", "node_iat_std",
       "in_degree", "out_degree"]  # topology
)
N_NODE_FEATURES = len(NODE_COL_ORDER)

EDGE_COL_ORDER: tuple[str, ...] = (
    "iat",
    *(f"{c}_diff" for c in BYTE_COLS),
    "bidir",
    "edge_freq",
)
N_EDGE_FEATURES = len(EDGE_COL_ORDER)


CAN_SCHEMA = GraphSchema(
    node_stat_exprs=DOMAIN_NODE_EXPRS + TOPOLOGY_NODE_PLACEHOLDERS,
    edge_stat_exprs=DOMAIN_EDGE_EXPRS,
    node_col_order=NODE_COL_ORDER,
    edge_col_order=EDGE_COL_ORDER,
    label_exprs=LABEL_EXPRS,
    edge_base_cols=BYTE_COLS,  # byte diffs need byte_0..7
    vocab_column="arb_id",
    attack_type_codes=ATTACK_TYPE_CODES,
    attack_type_names=ATTACK_TYPE_NAMES,
)


# ─── §6. Payload parser ───────────────────────────────────────────────

def parse_payload(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Hex 16-char ``payload`` → ``byte_0..7`` (Float32) + Shannon ``entropy``.
    Idempotent: passes through if ``byte_0`` already present.
    """
    if "byte_0" in lf.collect_schema().names():
        return lf
    byte_exprs = [
        pl.col("payload").str.slice(i * 2, 2).str.to_integer(base=16, strict=False)
        .fill_null(0).cast(pl.Float32).alias(f"byte_{i}")
        for i in range(N_BYTES)
    ]
    lf = lf.with_columns(byte_exprs)
    bcols = [pl.col(c) for c in BYTE_COLS]
    row_sum = pl.sum_horizontal(bcols).clip(1e-12, None)
    entropy = pl.sum_horizontal(
        [pl.when(c > 0).then(-(c / row_sum) * (c / row_sum).log()).otherwise(0.0) for c in bcols]
    ).alias("entropy")
    return lf.with_columns(entropy)


# ─── §7. Adapter classes ──────────────────────────────────────────────

class CANBusDataset(BaseGraphDataset):
    """One graph = one sliding window of CAN messages.

    Nodes are arbitration IDs; edges are temporal adjacency (shift-1).
    Only override is ``_read_raw``: lazy-scan declared source_dirs,
    parse hex, tag attack_type from filename.
    """

    SCHEMA: ClassVar[GraphSchema] = CAN_SCHEMA

    def _read_raw(self) -> pl.DataFrame:
        if not self.source_dirs:
            raise ValueError(
                f"CANBusDataset split={self.split!r} has no source_dirs; "
                "caller must pass source_dirs=[...] (recursive raw_data_dir scan "
                "would mix train+test → contamination)"
            )
        frames: list[pl.LazyFrame] = []
        for sub in self.source_dirs:
            sub_path = self.raw_data_dir / sub
            if not sub_path.is_dir():
                raise FileNotFoundError(f"declared source_dir {sub!r} missing under {self.raw_data_dir}")
            for csv_path in sorted(sub_path.rglob("*.csv")):
                at = self._infer_attack_type(csv_path)
                frames.append(pl.scan_csv(csv_path).with_columns(pl.lit(at).alias("attack_type")))
        if not frames:
            raise ValueError(f"no CSVs under any of {self.source_dirs!r} in {self.raw_data_dir}")

        combined = pl.concat(frames).sort("timestamp")
        # HCRL CSVs use different names — alias to schema vocabulary.
        cols = combined.collect_schema().names()
        renames = {old: new for old, new in (("arbitration_id", "arb_id"), ("data_field", "payload"))
                   if old in cols}
        if renames:
            combined = combined.rename(renames)
        return parse_payload(combined).collect()


@dataclass(frozen=True)
class CANBusSource(BaseGraphSource):
    """Catalog → vocab-once → train/val/per-test-subdir CANBusDataset."""

    KIND: ClassVar[str] = "canbus"
    DATASET_CLS: ClassVar[type[BaseGraphDataset]] = CANBusDataset

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        from graphids.core.data.preprocessing.vocab import scan_arb_ids

        return scan_arb_ids(raw_dir, source_dirs)
