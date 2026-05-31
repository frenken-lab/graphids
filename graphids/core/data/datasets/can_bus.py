"""CAN bus dataset adapter and schema."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Literal

import polars as pl
from structlog import get_logger

from graphids.core.data.datasets._base import (
    BaseGraphDataset,
    BaseGraphSource,
    GraphSchema,
)
from graphids.core.data.discovery.hypotheses import DiscoveryStore
from graphids.core.data.preprocessing.edge_policy import temporal_edge_policy
from graphids.core.data.preprocessing.graph_ops import default_graph_transforms
from graphids.core.data.preprocessing.representations import (
    TemporalRepresentationCfg,
    representation_digest,
    representation_temporal_spec,
)
from graphids.core.data.preprocessing.temporal import build_temporal_data
from graphids.core.data.preprocessing.transforms import TOPOLOGY_NODE_PLACEHOLDER_EXPRS
from graphids.core.data.state import DatasetState

log = get_logger(__name__)


N_BYTES = 8
BYTE_COLS = [f"byte_{i}" for i in range(N_BYTES)]

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

DOMAIN_EDGE_EXPRS: list[pl.Expr] = [
    pl.col("timestamp").diff().cast(pl.Float32).alias("iat"),
    *[pl.col(c).diff().abs().cast(pl.Float32).alias(f"{c}_diff") for c in BYTE_COLS],
]

LABEL_EXPRS: list[pl.Expr] = [
    (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
    pl.col("attack_type")
    .filter(pl.col("attack_type") > 0)
    .mode()
    .first()
    .fill_null(0)
    .alias("attack_type"),
]

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

CAN_EDGE_POLICY = temporal_edge_policy(src_col="node_id", dst_col="node_id", dst_shift=-1)
CAN_GRAPH_TRANSFORMS = tuple(default_graph_transforms())


CAN_SCHEMA = GraphSchema(
    node_stat_exprs=DOMAIN_NODE_EXPRS + TOPOLOGY_NODE_PLACEHOLDER_EXPRS,
    edge_stat_exprs=DOMAIN_EDGE_EXPRS,
    node_col_order=NODE_COL_ORDER,
    edge_col_order=EDGE_COL_ORDER,
    label_exprs=LABEL_EXPRS,
    edge_base_cols=BYTE_COLS,  # byte diffs need byte_0..7
    vocab_column="arb_id",
    attack_type_codes=ATTACK_TYPE_CODES,
    attack_type_names=ATTACK_TYPE_NAMES,
    edge_policy=CAN_EDGE_POLICY,
    graph_transforms=CAN_GRAPH_TRANSFORMS,
)

# Backward-compatible schema aliases used by tests and downstream configs.
NODE_STAT_EXPRS = CAN_SCHEMA.node_stat_exprs
EDGE_STAT_EXPRS = CAN_SCHEMA.edge_stat_exprs
EDGE_BASE_COLS = CAN_SCHEMA.edge_base_cols

def parse_payload(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Hex ``payload`` to ``byte_0..7`` plus Shannon entropy."""
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


def infer_attack_type(csv: Path) -> int:
    """Infer the attack code from filename/path substrings."""
    s = csv.stem.lower() + " " + csv.parent.name.lower()
    for kw, code in ATTACK_TYPE_CODES.items():
        if kw in s:
            return code
    return 0


def load_can_rows(raw_dir: Path, source_dirs: list[str]) -> pl.DataFrame:
    """Load, normalize, and parse raw CAN CSVs from source dirs."""
    if not source_dirs:
        raise ValueError("source_dirs is empty; cannot load CAN rows")
    frames: list[pl.LazyFrame] = []
    for sub in source_dirs:
        sub_path = raw_dir / sub
        if not sub_path.is_dir():
            raise FileNotFoundError(f"declared source_dir {sub!r} missing under {raw_dir}")
        for csv_path in sorted(sub_path.rglob("*.csv")):
            at = infer_attack_type(csv_path)
            frames.append(
                pl.scan_csv(csv_path).with_columns(
                    pl.lit(at).alias("attack_type"),
                    pl.lit(sub).alias("vehicle_id"),
                )
            )
    if not frames:
        raise ValueError(f"no CSVs under any of {source_dirs!r} in {raw_dir}")

    combined = pl.concat(frames).sort("timestamp")
    cols = combined.collect_schema().names()
    renames = {
        old: new
        for old, new in (("arbitration_id", "arb_id"), ("data_field", "payload"))
        if old in cols
    }
    if renames:
        combined = combined.rename(renames)
    return parse_payload(combined).collect()


class CANBusDataset(BaseGraphDataset):
    """One graph is one sliding window of CAN messages."""

    SCHEMA: ClassVar[GraphSchema] = CAN_SCHEMA

    def _read_raw(self) -> pl.DataFrame:
        if not self.source_dirs:
            raise ValueError(
                f"CANBusDataset split={self.split!r} has no source_dirs; "
                "caller must pass source_dirs=[...]"
            )
        return load_can_rows(self.raw_data_dir, self.source_dirs)


@dataclass(frozen=True)
class CANBusSource(BaseGraphSource):
    """Catalog to train/val/test CANBusDataset builder."""

    KIND: ClassVar[str] = "canbus"
    DATASET_CLS: ClassVar[type[BaseGraphDataset]] = CANBusDataset

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        from graphids.core.data.preprocessing.vocab import scan_arb_ids

        return scan_arb_ids(raw_dir, source_dirs)

    def _post_build_artifacts(
        self,
        *,
        root: Path,
        raw: Path,
        train_dirs: list[str],
        present_test: list[str],
        vocab: dict[str, int],
        digest: str,
    ) -> None:
        from graphids.core.data.discovery.hypotheses import (
            build_signal_profiles,
            initialize_hypotheses,
        )

        del present_test, vocab, digest
        profiles = build_signal_profiles(load_can_rows(raw, train_dirs))
        store = DiscoveryStore(root=Path(root))
        hypotheses = initialize_hypotheses(profiles)
        store.write_profiles(profiles)
        store.write_hypotheses(hypotheses)
        store.write_manifest(profiles=profiles, hypotheses=hypotheses)


@dataclass(frozen=True)
class TemporalCANBusSource:
    """Catalog to build temporal CAN event streams."""

    KIND: ClassVar[str] = "canbus_temporal"
    name: str
    seed: int
    lake_root: str | None = None
    val_fraction: float = 0.2
    vocab_scope: Literal["train", "all"] = "train"
    representation_cfg: TemporalRepresentationCfg = field(default_factory=TemporalRepresentationCfg)

    def resolved_lake_root(self) -> str:
        if self.lake_root:
            return self.lake_root
        from graphids.paths import lake_root

        return lake_root()

    @property
    def cache_key(self) -> str:
        return (
            f"{self.KIND}|{self.resolved_lake_root()}|{self.name}"
            f"|v{self.val_fraction}|seed{self.seed}|voc:{self.vocab_scope}"
            f"|repr:{self.representation_cfg.kind}:{representation_digest(self.representation_cfg)}"
        )

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        from graphids.core.data.preprocessing.vocab import scan_arb_ids

        return scan_arb_ids(raw_dir, source_dirs)

    def build(self) -> DatasetState:
        from graphids.core.data.preprocessing.vocab import persist_vocab
        from graphids.paths import cache_dir, data_dir, load_catalog

        entry = load_catalog()[self.name]
        lake = self.resolved_lake_root()
        raw = data_dir(lake, self.name)
        root = (
            cache_dir(lake, self.name)
            / f"temporal_{representation_digest(self.representation_cfg)}_voc_{self.vocab_scope}"
        )

        train_dirs = [
            s for s in (entry.get("train_subdir"), entry.get("train_attack_subdir")) if s
        ]
        if not train_dirs:
            raise ValueError(f"catalog entry {self.name!r} declares no train_subdir(s)")

        present_test = [sd for sd in entry.get("test_subdirs", []) if (raw / sd).is_dir()]
        scan_sources = list(train_dirs) + (present_test if self.vocab_scope == "all" else [])
        vocab = {tok: i + 1 for i, tok in enumerate(self._scan_vocab(raw, scan_sources))}
        Path(root).mkdir(parents=True, exist_ok=True)
        persist_vocab(vocab, Path(root) / "vocab.json")

        def _encode(df: pl.DataFrame) -> pl.DataFrame:
            return df.with_columns(
                pl.col("arb_id").replace_strict(vocab, default=0).cast(pl.Int64).alias("node_id")
            )

        spec = replace(
            representation_temporal_spec(self.representation_cfg),
            time_col=self.representation_cfg.time_col,
            feature_cols=tuple(BYTE_COLS) + ("entropy",),
            target_col="attack",
            aux_label_cols=("attack_type",),
        )

        train_rows = _encode(load_can_rows(raw, train_dirs)).sort("timestamp")
        n_rows = len(train_rows)
        n_val = int(n_rows * self.val_fraction)
        train_td = build_temporal_data(train_rows.head(max(0, n_rows - n_val)), spec)
        val_td = build_temporal_data(train_rows.tail(n_val), spec)

        tests = {
            sd: build_temporal_data(_encode(load_can_rows(raw, [sd])).sort("timestamp"), spec)
            for sd in present_test
        }
        return DatasetState(train=train_td, val=val_td, test=tests)


__all__ = [
    "ATTACK_TYPE_CODES",
    "ATTACK_TYPE_NAMES",
    "BYTE_COLS",
    "NODE_STAT_EXPRS",
    "EDGE_STAT_EXPRS",
    "EDGE_BASE_COLS",
    "LABEL_EXPRS",
    "CAN_SCHEMA",
    "CANBusDataset",
    "CANBusSource",
    "TemporalCANBusSource",
    "parse_payload",
    "infer_attack_type",
    "load_can_rows",
]
