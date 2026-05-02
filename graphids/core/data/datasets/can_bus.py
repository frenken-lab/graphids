"""CAN bus dataset — I/O, vocab, and feature schema.

Everything CAN-bus-specific lives here: hex payload parsing, byte-column
feature expressions, attack-type taxonomy, and the ``CANBusDataset`` +
``CANBusSource`` adapters. The general sliding-window pipeline lives in
``graph_pipeline.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from filelock import FileLock
from torch_geometric.data import Data, InMemoryDataset

from graphids._fs import atomic_save
from structlog import get_logger
from graphids.config.constants import PREPROCESSING_VERSION
from graphids.core.data import scaler as scaler_mod
from graphids.core.data.cache import DatasetState
from graphids.core.data.graph_pipeline import GraphPipeline
from graphids.core.data.metadata import merge_split_into_metadata

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Attack-type taxonomy (CAN bus specific)
# ---------------------------------------------------------------------------

ATTACK_TYPE_CODES: dict[str, int] = {
    "normal": 0,
    "attack_free": 0,
    "benign": 0,
    "dos": 1,
    "fuzzy": 2,
    "fuzzing": 2,
    "gear": 3,
    "rpm": 4,
    "flooding": 5,
    "malfunction": 6,
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


def _describe(t: torch.Tensor) -> dict[str, float | int]:
    """min/max/mean/p95/p99 of a 1-D tensor — the cache_metadata stat block."""
    return {
        "min": int(t.min().item()),
        "max": int(t.max().item()),
        "mean": float(t.mean().item()),
        "p95": float(t.quantile(0.95).item()),
        "p99": float(t.quantile(0.99).item()),
    }


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
# CANBusDataset — InMemoryDataset adapter
# ---------------------------------------------------------------------------


class CANBusDataset(InMemoryDataset):
    """CAN bus intrusion detection graph dataset.

    Each graph is one sliding window of CAN messages. Nodes are arbitration
    IDs, edges are temporal adjacency (shift-1).
    """

    def __init__(
        self,
        root: str | Path,
        raw_dir: str | Path,
        *,
        val_fraction: float,
        split: str = "train",
        source_dirs: list[str] | None = None,
        split_tag: str | None = None,
        window_size: int = 50,
        stride: int = 25,
        seed: int = 42,
        shared_vocab: dict | None = None,
        shared_vocab_digest: str | None = None,
        scaler_strategy: str = "z_benign",
        transform=None,
        pre_transform=None,
    ):
        self.raw_data_dir = Path(raw_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.source_dirs = source_dirs
        # train/val share one tensor ("data_train.pt"); test splits are
        # one tensor per subdir (caller passes split_tag="test_<subdir>").
        if split_tag is None:
            if split in ("train", "val"):
                split_tag = "train"
            else:
                raise ValueError(f"split_tag is required for split={split!r}")
        self.split_tag = split_tag
        self.window_size = window_size
        self.stride = stride
        self.seed = seed
        self._shared_vocab = shared_vocab
        self._shared_vocab_digest = shared_vocab_digest
        self.scaler_strategy = scaler_strategy
        super().__init__(str(root), transform, pre_transform)
        self.load(self.processed_paths[0])
        self._load_num_arb_ids()

        if self.split in ("train", "val"):
            self._apply_train_val_split()

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{self.split_tag}.pt"]

    def _load_num_arb_ids(self) -> None:
        # Source of truth is ``cache_metadata.json`` (written by
        # ``merge_split_into_metadata`` from the shared-vocab size). The
        # old ``node_id.max() + 1`` fallback under-reported ``num_arb_ids``
        # when a split didn't contain every arb_id, causing the model's
        # embedding table to be under-sized and crashing at test time
        # with IndexError. See ``~/plans/oov-embedding-handling.md``.
        from graphids.core.data.metadata import load_metadata

        meta = load_metadata(Path(self.root))
        self.num_arb_ids = int(meta["num_arb_ids"])

    # ── Size tensors for NodeBudgetBatchSampler ───────────────────────
    # Derived from slices at zero I/O cost — the slice tensors are small
    # cumulative offsets (one int64 per graph + 1, ≈400KB for 50K graphs).
    # This lets the sampler walk sizes without reconstructing Data objects
    # per graph per epoch, which was 50K mmap reconstructions per epoch
    # under PyG's DynamicBatchSampler.__iter__.

    @property
    def num_nodes_per_graph(self) -> torch.Tensor:
        """Per-graph node counts for the current split (respects _indices)."""
        full = self.slices["x"][1:] - self.slices["x"][:-1]
        if self._indices is None:
            return full
        return full[torch.as_tensor(list(self._indices), dtype=torch.long)]

    @property
    def num_edges_per_graph(self) -> torch.Tensor:
        """Per-graph edge counts for the current split (respects _indices)."""
        full = self.slices["edge_index"][1:] - self.slices["edge_index"][:-1]
        if self._indices is None:
            return full
        return full[torch.as_tensor(list(self._indices), dtype=torch.long)]

    def _apply_train_val_split(self) -> None:
        n = len(self)
        gen = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n, generator=gen)
        n_val = int(n * self.val_fraction)
        self._indices = (perm[:n_val] if self.split == "val" else perm[n_val:]).tolist()

    # ── NFS-safe overrides ────────────────────────────────────────────

    def load(self, path: str, **kwargs):
        (self.data, self.slices) = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )

    def process(self) -> None:
        lock_path = Path(self.processed_dir) / ".lock"
        with FileLock(str(lock_path)):
            marker = Path(self.processed_dir) / ".complete"
            if Path(self.processed_paths[0]).exists() and marker.exists():
                return
            data, slices, num_arb_ids, num_graphs, num_raw = self._build_graphs()
            # Standardize x + edge_attr. Train fits via the configured
            # strategy (see graphids.core.data.scaler.STRATEGIES) on
            # train-only indices — same seeded permutation as
            # _apply_train_val_split so val rows don't leak. Test loads
            # the saved sklearn estimators.
            scaler_path = Path(self.processed_dir) / "feature_scaler.pt"
            if self.split == "train":
                gen = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(num_graphs, generator=gen)
                train_idx = perm[int(num_graphs * self.val_fraction) :]
                scaler = scaler_mod.fit(data, slices, train_idx, strategy=self.scaler_strategy)
                torch.save(scaler, scaler_path)
            else:
                if not scaler_path.exists():
                    raise FileNotFoundError(
                        f"feature_scaler.pt missing at {scaler_path}; "
                        "build the 'train' split before any 'test' split"
                    )
                scaler = torch.load(scaler_path, map_location="cpu", weights_only=False)
            scaler_mod.apply(data, scaler)
            tensor_path = Path(self.processed_paths[0])
            atomic_save([data, slices], tensor_path)
            (Path(self.processed_dir) / "num_arb_ids.txt").write_text(str(num_arb_ids))

            bytes_on_disk = tensor_path.stat().st_size
            dataset_name = Path(self.root).name
            invariants = {
                "preprocessing_version": PREPROCESSING_VERSION,
                "window_size": self.window_size,
                "stride": self.stride,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "vocab_digest": self._shared_vocab_digest,
                "scaler_strategy": self.scaler_strategy,
            }

            if self.split == "train":
                # Deterministic train/val partition (mirrors
                # _apply_train_val_split). We write both entries here so the
                # single train build fully populates the metadata without
                # needing val to re-enter process().
                gen = torch.Generator().manual_seed(self.seed)
                perm = torch.randperm(num_graphs, generator=gen)
                n_val = int(num_graphs * self.val_fraction)
                val_idx = perm[:n_val]
                train_idx = perm[n_val:]

                train_entry = self._build_split_entry(
                    data,
                    slices,
                    indices=train_idx,
                    num_raw_samples=num_raw,
                    bytes_on_disk=bytes_on_disk,
                    source_dirs=self.source_dirs,
                )
                # val shares train's tensor — entry is minimal (no graph_stats
                # or per-split raw_samples; consumers that want a val
                # distribution must slice the train tensor themselves).
                val_entry = {
                    "num_graphs": int(val_idx.numel()),
                    "derived_from": "train",
                    "val_fraction_seed": [self.val_fraction, self.seed],
                }
                merge_split_into_metadata(
                    Path(self.root),
                    "train",
                    train_entry,
                    invariants=invariants,
                    dataset_name=dataset_name,
                    num_arb_ids=num_arb_ids,
                )
                merge_split_into_metadata(
                    Path(self.root),
                    "val",
                    val_entry,
                    invariants=invariants,
                    dataset_name=dataset_name,
                    num_arb_ids=num_arb_ids,
                )
            else:  # split == "test": one tensor = one test subdir
                test_entry = self._build_split_entry(
                    data,
                    slices,
                    indices=None,
                    num_raw_samples=num_raw,
                    bytes_on_disk=bytes_on_disk,
                    source_dirs=self.source_dirs,
                )
                merge_split_into_metadata(
                    Path(self.root),
                    self.split_tag,
                    test_entry,
                    invariants=invariants,
                    dataset_name=dataset_name,
                    num_arb_ids=num_arb_ids,
                )
            marker.write_text("ok")

    def _build_split_entry(
        self,
        data: Data,
        slices: dict,
        *,
        indices: torch.Tensor | None = None,
        num_raw_samples: int | None = None,
        bytes_on_disk: int | None = None,
        source_dirs: list[str] | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Compose a per-split metadata entry from graph tensors.

        ``indices`` (when given) scopes stats + attack balance to a
        post-split subset — used so ``splits.train`` / ``splits.val``
        report their own slice of the shared train tensor.
        """
        node_diffs = (slices["x"][1:] - slices["x"][:-1]).float()
        edge_diffs = (slices["edge_index"][1:] - slices["edge_index"][:-1]).float()
        attack_types = data.attack_type
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long)
            node_t = node_diffs.index_select(0, idx)
            edge_t = edge_diffs.index_select(0, idx)
            attack_types = attack_types.index_select(0, idx)
        else:
            node_t = node_diffs
            edge_t = edge_diffs

        balance: dict[str, int] = {}
        for t in attack_types.tolist():
            name = ATTACK_TYPE_NAMES.get(int(t), f"unknown_{int(t)}")
            balance[name] = balance.get(name, 0) + 1

        entry: dict = {
            "num_graphs": int(node_t.numel()),
            "graph_stats": {
                "node_count": _describe(node_t),
                "edge_count": _describe(edge_t),
            },
            "attack_balance": balance,
        }
        if source_dirs is not None:
            entry["source_dirs"] = list(source_dirs)
        if num_raw_samples is not None:
            entry["num_raw_samples"] = int(num_raw_samples)
        if bytes_on_disk is not None:
            entry["bytes_on_disk"] = int(bytes_on_disk)
        if extra:
            entry.update(extra)
        return entry

    # ── pipeline ──────────────────────────────────────────────────────

    def _build_graphs(self) -> tuple[Data, dict, int, int, int]:
        df = self._read_raw()
        log.info("raw_loaded", rows=len(df))

        if self._shared_vocab is None:
            raise ValueError(
                f"CANBusDataset cannot build cache for split={self.split!r} without "
                f"shared_vocab. Construct via CANBusSource.build(), which scans all "
                f"splits' source_dirs and persists a shared vocab. "
                f"Root: {self.root}"
            )
        vocab = self._shared_vocab
        num_arb_ids = len(vocab) + 1  # +1 for UNK at index 0
        df = df.with_columns(
            pl.col("arb_id").replace_strict(vocab, default=0).cast(pl.Int64).alias("node_id")
        )

        pipeline = GraphPipeline(
            node_stat_exprs=NODE_STAT_EXPRS,
            edge_stat_exprs=EDGE_STAT_EXPRS,
            node_col_order=NODE_COL_ORDER,
            edge_col_order=EDGE_COL_ORDER,
            label_exprs=LABEL_EXPRS,
            edge_base_cols=EDGE_BASE_COLS,
        )
        data, slices, num_graphs, num_raw_samples = pipeline.run(
            df,
            self.window_size,
            self.stride,
        )
        del df
        return data, slices, num_arb_ids, num_graphs, num_raw_samples

    @staticmethod
    def _infer_attack_type(csv_path: Path) -> int:
        """Infer attack type code from file/directory naming conventions."""
        parts = csv_path.stem.lower() + " " + csv_path.parent.name.lower()
        for keyword, code in ATTACK_TYPE_CODES.items():
            if keyword in parts:
                return code
        return 0

    def _read_raw(self) -> pl.DataFrame:
        """Lazy-scan CSVs from declared source_dirs, parse hex, tag attack types.

        Scope is explicit: only subdirs in ``self.source_dirs`` are read.
        Recursive glob over ``raw_data_dir`` would silently pull every
        train+test CSV into one tensor (contamination; see plan §1.2).
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
# CANBusSource — dataset source wrapper for the process-level cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CANBusSource:
    """CAN bus dataset source — produces train/val/test splits on demand.

    ``get_or_build`` in ``graphids.core.data.cache`` memoizes the
    ``DatasetState`` returned by ``build()`` under ``cache_key`` so
    multi-stage runs sharing a process pay preprocessing + mmap cost
    once instead of per-stage.

    ``name`` is a catalog entry (e.g. ``hcrl_sa``, ``set_01``). The
    catalog is loaded at build time via
    ``graphids.config.catalog.load_catalog`` — no name validation at
    construction, since the catalog may shift.
    """

    name: str
    lake_root: str | None = None
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2
    seed: int = 42
    scaler_strategy: str = "z_benign"

    def resolved_lake_root(self) -> str:
        """Return ``lake_root`` falling back to the global settings value."""
        if self.lake_root:
            return self.lake_root
        from graphids.config.settings import get_settings

        return get_settings().lake_root

    @property
    def cache_key(self) -> str:
        return (
            f"canbus|{self.resolved_lake_root()}|{self.name}"
            f"|w{self.window_size}|s{self.stride}"
            f"|v{self.val_fraction}|seed{self.seed}"
            f"|sc:{self.scaler_strategy}"
        )

    def build(self) -> DatasetState:
        from graphids.config.catalog import cache_dir, data_dir, load_catalog

        entry = load_catalog()[self.name]
        lake_root = self.resolved_lake_root()
        root = cache_dir(lake_root, self.name)
        raw = data_dir(lake_root, self.name)

        # Train scope is explicit: attack-free + with-attacks subdirs from
        # the catalog. Missing fields are skipped so datasets without a
        # with-attacks split still work.
        train_dirs = [s for s in (entry.get("train_subdir"), entry.get("train_attack_subdir")) if s]
        if not train_dirs:
            raise ValueError(
                f"Catalog entry for {self.name!r} declares no train_subdir "
                f"or train_attack_subdir; cannot build training cache."
            )

        # Shared vocab: scanned once across train + every present test
        # subdir so every split maps an arb_id to the same embedding row.
        # Persisted under {root}/vocab.json; its digest becomes a cache
        # invariant (see ``metadata.INVARIANT_KEYS``) so adding a subdir
        # with new arb_ids forces a clean rebuild.
        from graphids.core.data.vocab import persist_vocab, scan_arb_ids

        present_test_subdirs = [sd for sd in entry.get("test_subdirs", []) if (raw / sd).is_dir()]
        all_sources = list(train_dirs) + present_test_subdirs
        # Dense index starting at 1; 0 reserved for UNK. scan_arb_ids
        # returns sorted uniques, so enumerate order is deterministic.
        shared_vocab = {tok: i + 1 for i, tok in enumerate(scan_arb_ids(raw, all_sources))}
        shared_vocab_digest = persist_vocab(shared_vocab, Path(root) / "vocab.json")

        common = dict(
            window_size=self.window_size,
            stride=self.stride,
            val_fraction=self.val_fraction,
            seed=self.seed,
            shared_vocab=shared_vocab,
            shared_vocab_digest=shared_vocab_digest,
            scaler_strategy=self.scaler_strategy,
        )
        train_ds = CANBusDataset(
            root=root,
            raw_dir=raw,
            split="train",
            source_dirs=train_dirs,
            split_tag="train",
            **common,
        )
        val_ds = CANBusDataset(
            root=root,
            raw_dir=raw,
            split="val",
            source_dirs=train_dirs,
            split_tag="train",
            **common,
        )

        # Per-test-subdir tensors: each subdir gets its own data_test_<name>.pt,
        # preventing the "all test_N eval against test_01" regression
        # (plan §1.3).
        test_datasets: dict[str, CANBusDataset] = {}
        for subdir in present_test_subdirs:
            test_datasets[subdir] = CANBusDataset(
                root=root,
                raw_dir=raw,
                split="test",
                source_dirs=[subdir],
                split_tag=f"test_{subdir}",
                **common,
            )
        return DatasetState(train=train_ds, val=val_ds, test=test_datasets)
