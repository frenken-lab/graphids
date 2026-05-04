"""Generic graph-dataset base + source.

Subclasses provide a :class:`GraphSchema` (Polars exprs + column orders +
attack taxonomy) and implement two hooks: ``_read_raw`` on the dataset
(loads raw files into a long-format frame) and ``_scan_vocab`` on the
source (returns the sorted unique vocab values across every CSV the
catalog declares). Everything else — train/val splitting, scaler fit/
apply, metadata merge, mmap load, per-test-subdir tensor layout — lives
here.

Cache layout invariants (see ``preprocessing/metadata.py``) are unchanged
across subclasses; the on-disk schema field is still ``num_arb_ids`` for
backward compatibility, so this lift does NOT bump
``METADATA_SCHEMA_VERSION``. Future schema generalization (``num_vocab_ids``)
is a separate change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import torch
from filelock import FileLock
from structlog import get_logger
from torch_geometric.data import Data, InMemoryDataset

from graphids._fs import atomic_save
from graphids.config.constants import PREPROCESSING_VERSION
from graphids.core.data.preprocessing import scaler as scaler_mod
from graphids.core.data.preprocessing.metadata import (
    load_metadata,
    merge_split_into_metadata,
)
from graphids.core.data.preprocessing.pipeline import GraphPipeline
from graphids.core.data.preprocessing.vocab import persist_vocab
from graphids.core.data.state import DatasetState

log = get_logger(__name__)


def _describe(t: torch.Tensor) -> dict[str, float | int]:
    """min/max/mean/p95/p99 of a 1-D tensor — the cache_metadata stat block."""
    return {
        "min": int(t.min().item()),
        "max": int(t.max().item()),
        "mean": float(t.mean().item()),
        "p95": float(t.quantile(0.95).item()),
        "p99": float(t.quantile(0.99).item()),
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSchema:
    """Per-domain feature/label schema consumed by GraphPipeline + cache writer.

    ``vocab_column`` is the raw-frame column whose unique values become the
    embedding vocabulary (e.g. ``arb_id`` for CAN, ``sensor_name`` for ICS).
    The column is replaced into ``node_id`` (Int64) at build time with
    ``replace_strict(vocab, default=0)``; index 0 is reserved for UNK.

    ``attack_type_codes`` / ``attack_type_names`` are optional. Datasets
    without a multiclass attack taxonomy can leave both ``None`` — the
    base falls back to ``{0: "benign"}`` for the per-split balance block,
    and ``_infer_attack_type`` returns 0 unconditionally.
    """

    node_stat_exprs: list[pl.Expr]
    edge_stat_exprs: list[pl.Expr]
    node_col_order: list[str]
    edge_col_order: tuple[str, ...]
    label_exprs: list[pl.Expr]
    edge_base_cols: list[str]
    vocab_column: str
    attack_type_codes: dict[str, int] | None = None
    attack_type_names: dict[int, str] | None = None


# ---------------------------------------------------------------------------
# BaseGraphDataset
# ---------------------------------------------------------------------------


class BaseGraphDataset(InMemoryDataset):
    """Sliding-window graph dataset with shared vocab + scaler + metadata.

    Subclasses set the class-level ``SCHEMA`` and implement ``_read_raw``.
    Public attributes match the original ``CANBusDataset`` for drop-in
    replacement: ``num_arb_ids`` (still the on-disk metadata field name),
    ``num_nodes_per_graph`` / ``num_edges_per_graph`` (size tensors for
    NodeBudgetBatchSampler), ``split``, ``split_tag``.
    """

    SCHEMA: ClassVar[GraphSchema]

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
        self._load_num_ids()

        if self.split in ("train", "val"):
            self._apply_train_val_split()

    # ── must override ─────────────────────────────────────────────────

    def _read_raw(self) -> pl.DataFrame:
        """Load raw files; return a long-format frame.

        The frame must contain ``timestamp`` + ``SCHEMA.vocab_column`` +
        every column referenced by ``SCHEMA.node_stat_exprs`` /
        ``edge_stat_exprs`` / ``label_exprs``.
        """
        raise NotImplementedError(
            f"{type(self).__name__}._read_raw must return a long-format pl.DataFrame"
        )

    # ── default behavior ──────────────────────────────────────────────

    def _infer_attack_type(self, csv_path: Path) -> int:
        """Match a CSV's filename + parent dir against SCHEMA.attack_type_codes.

        Override if a domain encodes attack type differently (column flag,
        sidecar manifest, etc.). Returns 0 when no taxonomy is configured.
        """
        codes = self.SCHEMA.attack_type_codes or {}
        parts = csv_path.stem.lower() + " " + csv_path.parent.name.lower()
        for kw, code in codes.items():
            if kw in parts:
                return code
        return 0

    # ── generic plumbing ──────────────────────────────────────────────

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{self.split_tag}.pt"]

    def _load_num_ids(self) -> None:
        # Source of truth is ``cache_metadata.json`` (written by
        # ``merge_split_into_metadata`` from the shared-vocab size). The
        # old ``node_id.max() + 1`` fallback under-reported when a split
        # didn't contain every vocab id, causing the model's embedding
        # table to be under-sized and crashing at test time with
        # IndexError. See ``~/plans/oov-embedding-handling.md``.
        # On-disk JSON key remains ``num_arb_ids`` for cache compatibility;
        # exposing it generically as ``self.num_ids`` keeps the
        # ``BaseGraphDataset`` runtime attribute domain-agnostic.
        meta = load_metadata(Path(self.root))
        self.num_ids = int(meta["num_arb_ids"])

    # Size tensors for NodeBudgetBatchSampler. Derived from slices at
    # zero I/O cost — the slice tensors are small cumulative offsets
    # (one int64 per graph + 1, ~400KB for 50K graphs). This lets the
    # sampler walk sizes without reconstructing Data objects per graph
    # per epoch.

    @property
    def num_nodes_per_graph(self) -> torch.Tensor:
        full = self.slices["x"][1:] - self.slices["x"][:-1]
        if self._indices is None:
            return full
        return full[torch.as_tensor(list(self._indices), dtype=torch.long)]

    @property
    def num_edges_per_graph(self) -> torch.Tensor:
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
                # _apply_train_val_split). Both entries written here so a
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

        names = self.SCHEMA.attack_type_names or {0: "benign"}
        balance: dict[str, int] = {}
        for t in attack_types.tolist():
            name = names.get(int(t), f"unknown_{int(t)}")
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
                f"{type(self).__name__} cannot build cache for split={self.split!r} "
                f"without shared_vocab. Construct via the source's build(), which "
                f"scans every split's source_dirs and persists a shared vocab. "
                f"Root: {self.root}"
            )
        vocab = self._shared_vocab
        num_arb_ids = len(vocab) + 1  # +1 for UNK at index 0
        df = df.with_columns(
            pl.col(self.SCHEMA.vocab_column)
            .replace_strict(vocab, default=0)
            .cast(pl.Int64)
            .alias("node_id")
        )

        pipeline = GraphPipeline(
            node_stat_exprs=self.SCHEMA.node_stat_exprs,
            edge_stat_exprs=self.SCHEMA.edge_stat_exprs,
            node_col_order=self.SCHEMA.node_col_order,
            edge_col_order=self.SCHEMA.edge_col_order,
            label_exprs=self.SCHEMA.label_exprs,
            edge_base_cols=self.SCHEMA.edge_base_cols,
        )
        data, slices, num_graphs, num_raw_samples = pipeline.run(
            df,
            self.window_size,
            self.stride,
        )
        del df
        return data, slices, num_arb_ids, num_graphs, num_raw_samples


# ---------------------------------------------------------------------------
# BaseGraphSource
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaseGraphSource:
    """Generic source: catalog → vocab-once → train/val/test datasets.

    Concrete subclasses set ``KIND`` (cache_key prefix) and ``DATASET_CLS``
    (the :class:`BaseGraphDataset` subclass to instantiate), and implement
    ``_scan_vocab`` (returns sorted unique vocab values across every CSV
    the catalog declares).

    Catalog shape this base assumes — ``train_subdir`` /
    ``train_attack_subdir`` / ``test_subdirs`` (the CAN-bus layout). A
    domain whose splits don't fit this mold (single-CSV WaDi, e.g.) can
    override ``build()`` directly.
    """

    KIND: ClassVar[str]
    DATASET_CLS: ClassVar[type[BaseGraphDataset]]

    name: str
    lake_root: str | None = None
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2
    seed: int = 42
    scaler_strategy: str = "z_benign"

    def resolved_lake_root(self) -> str:
        if self.lake_root:
            return self.lake_root
        from graphids.config.catalog import lake_root

        return lake_root()

    @property
    def cache_key(self) -> str:
        return (
            f"{self.KIND}|{self.resolved_lake_root()}|{self.name}"
            f"|w{self.window_size}|s{self.stride}"
            f"|v{self.val_fraction}|seed{self.seed}"
            f"|sc:{self.scaler_strategy}"
        )

    # ── must override ─────────────────────────────────────────────────

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        """Return sorted unique values of ``SCHEMA.vocab_column`` across
        every CSV under every source_dir.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override _scan_vocab() to return "
            "sorted unique values of SCHEMA.vocab_column across all source_dirs."
        )

    # ── generic plumbing ──────────────────────────────────────────────

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
        # subdir so every split maps an id to the same embedding row.
        # Persisted under {root}/vocab.json; its digest becomes a cache
        # invariant (see ``metadata.INVARIANT_KEYS``) so adding a subdir
        # with new ids forces a clean rebuild.
        present_test_subdirs = [sd for sd in entry.get("test_subdirs", []) if (raw / sd).is_dir()]
        all_sources = list(train_dirs) + present_test_subdirs
        # Dense index starting at 1; 0 reserved for UNK. _scan_vocab
        # returns sorted uniques, so enumerate order is deterministic.
        shared_vocab = {tok: i + 1 for i, tok in enumerate(self._scan_vocab(raw, all_sources))}
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
        train_ds = self.DATASET_CLS(
            root=root,
            raw_dir=raw,
            split="train",
            source_dirs=train_dirs,
            split_tag="train",
            **common,
        )
        val_ds = self.DATASET_CLS(
            root=root,
            raw_dir=raw,
            split="val",
            source_dirs=train_dirs,
            split_tag="train",
            **common,
        )

        # Per-test-subdir tensors: each subdir gets its own
        # data_test_<name>.pt, preventing the "all test_N eval against
        # test_01" regression.
        test_datasets: dict[str, BaseGraphDataset] = {}
        for subdir in present_test_subdirs:
            test_datasets[subdir] = self.DATASET_CLS(
                root=root,
                raw_dir=raw,
                split="test",
                source_dirs=[subdir],
                split_tag=f"test_{subdir}",
                **common,
            )
        return DatasetState(train=train_ds, val=val_ds, test=test_datasets)
