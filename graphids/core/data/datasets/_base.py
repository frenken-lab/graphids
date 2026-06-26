"""Base dataset and source primitives for graph preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import polars as pl
import torch
from filelock import FileLock
from torch_geometric.data import Data, InMemoryDataset

from graphids._fs import atomic_save, atomic_write_text
from graphids.core.data.preprocessing import scaler as scaler_mod
from graphids.core.data.preprocessing.materialization import build_graph_tables
from graphids.core.data.preprocessing.pyg import graph_tables_to_pyg
from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_window_defaults,
)
from graphids.core.data.preprocessing.scaler import (
    ScalerCfg,
    ZBenignScalerCfg,
    scaler_kind,
)
from graphids.core.data.preprocessing.splits import (
    split_embargo_width,
    split_graph_indices,
)
from graphids.core.data.preprocessing.vocab import persist_vocab
from graphids.core.data.state import DatasetState

_DEFAULT_SCALER_CFG = ZBenignScalerCfg()
_DEFAULT_REPRESENTATION_CFG = SnapshotRepresentationCfg()


@dataclass(frozen=True)
class GraphSchema:
    node_stat_exprs: list[pl.Expr]
    edge_stat_exprs: list[pl.Expr]
    node_col_order: list[str]
    edge_col_order: tuple[str, ...]
    label_exprs: list[pl.Expr]
    edge_base_cols: list[str]
    vocab_column: str


class BaseGraphDataset(InMemoryDataset):
    """Sliding-window graph InMemoryDataset with shared vocab + scaler.

    Subclass: set ``SCHEMA`` and implement ``_read_raw`` returning a
    long-format polars DataFrame containing ``timestamp`` + the
    schema's ``vocab_column`` + every column referenced by the exprs.
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
        shared_vocab: dict | None = None,
        shared_vocab_digest: str | None = None,
        vocab_scope: Literal["train", "all"] = "train",
        scaler_cfg: ScalerCfg = _DEFAULT_SCALER_CFG,
        representation_cfg: GraphRepresentationCfg = _DEFAULT_REPRESENTATION_CFG,
        transform=None,
        pre_transform=None,
    ):
        self.raw_data_dir = Path(raw_dir)
        self.split = split
        self.val_fraction = val_fraction
        self.source_dirs = source_dirs
        self._shared_vocab = shared_vocab
        self._shared_vocab_digest = shared_vocab_digest
        self.vocab_scope = vocab_scope
        self.scaler_cfg = scaler_cfg
        self.scaler_strategy = scaler_kind(scaler_cfg)
        self.representation_cfg = representation_cfg
        self.representation_kind = representation_kind(representation_cfg)
        self.window_size, self.stride = representation_window_defaults(representation_cfg)
        force_reload = self._cache_is_vocab_stale(Path(root))
        super().__init__(str(root), transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0])
        self.num_ids = int(getattr(self._data, "num_ids", len(shared_vocab or {}) + 1))
        if self.split in ("train", "val"):
            train_idx, val_idx = split_graph_indices(
                self._data,
                self.slices,
                self.representation_cfg,
                val_fraction=self.val_fraction,
            )
            idx = val_idx if self.split == "val" else train_idx
            self._indices = idx.tolist()

    # ── must override ──────────────────────────────────────────────────
    def _read_raw(self) -> pl.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__}._read_raw must return a long-format pl.DataFrame"
        )

    @property
    def cache_split_name(self) -> str:
        if self.split in ("train", "val"):
            return "train"
        if self.split == "test":
            if not self.source_dirs or len(self.source_dirs) != 1:
                raise ValueError(
                    f"split={self.split!r} needs exactly one source_dir"
                )
            return f"test_{self.source_dirs[0]}"
        return self.split

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_{self.cache_split_name}.pt"]

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

    # ── NFS-safe overrides ─────────────────────────────────────────────
    def load(self, path: str, **kwargs):
        self.data, self.slices = torch.load(
            path, map_location="cpu", mmap=True, weights_only=False
        )

    def _vocab_digest_path(self, tensor_path: Path) -> Path:
        return tensor_path.with_suffix(tensor_path.suffix + ".vocab_digest")

    def _cache_vocab_matches(self, tensor_path: Path) -> bool:
        if self._shared_vocab_digest is None:
            return True
        digest_path = self._vocab_digest_path(tensor_path)
        return digest_path.exists() and digest_path.read_text().strip() == self._shared_vocab_digest

    def _cache_is_vocab_stale(self, root: Path) -> bool:
        if self._shared_vocab_digest is None:
            return False
        tensor_path = root / "processed" / self.processed_file_names[0]
        return tensor_path.exists() and not self._cache_vocab_matches(tensor_path)

    def process(self) -> None:
        with FileLock(str(Path(self.processed_dir) / ".lock")):
            marker = Path(self.processed_dir) / ".complete"
            tensor_path = Path(self.processed_paths[0])
            if tensor_path.exists() and marker.exists() and self._cache_vocab_matches(tensor_path):
                return

            data, slices, _num_raw = self._build_graphs()

            scaler_path = Path(self.processed_dir) / "feature_scaler.pt"
            split_indices = (
                split_graph_indices(
                    data,
                    slices,
                    self.representation_cfg,
                    val_fraction=self.val_fraction,
                )
                if self.split in ("train", "val")
                else None
            )

            if self.split == "train":
                if split_indices is None:
                    raise RuntimeError("train split missing split indices")
                train_idx, _val_idx = split_indices
                scaler = scaler_mod.fit_from_cfg(data, slices, train_idx, cfg=self.scaler_cfg)
                torch.save(scaler, scaler_path)
            else:
                if not scaler_path.exists():
                    raise FileNotFoundError(
                        f"feature_scaler.pt missing at {scaler_path}; build train first"
                    )
                scaler = torch.load(scaler_path, map_location="cpu", weights_only=False)
            scaler_mod.apply(data, scaler)
            atomic_save([data, slices], tensor_path)
            if self._shared_vocab_digest is not None:
                atomic_write_text(
                    self._vocab_digest_path(tensor_path),
                    self._shared_vocab_digest + "\n",
                )
            marker.write_text("ok")

    def _build_graphs(self) -> tuple[Data, dict, int]:
        df = self._read_raw()
        if self._shared_vocab is None:
            raise ValueError(
                f"{type(self).__name__} needs shared_vocab for split={self.split!r}; "
                "build via the source's build() so vocab is scanned across splits"
            )
        vocab = self._shared_vocab
        df = df.with_columns(
            pl.col(self.SCHEMA.vocab_column)
            .replace_strict(vocab, default=0)
            .cast(pl.Int64)
            .alias("node_id")
        )
        return self._build_graphs_from_df(df, len(vocab) + 1)

    def _build_graphs_from_df(self, df: pl.DataFrame, num_ids: int) -> tuple[Data, dict, int]:
        tables = build_graph_tables(
            df,
            node_stat_exprs=self.SCHEMA.node_stat_exprs,
            label_exprs=self.SCHEMA.label_exprs,
            edge_stat_exprs=self.SCHEMA.edge_stat_exprs,
            edge_base_cols=self.SCHEMA.edge_base_cols,
            representation_cfg=self.representation_cfg,
        )
        del df
        if tables.node_stats.is_empty():
            return Data(num_ids=num_ids), {}, tables.n_rows
        data, slices, num_graphs, num_raw = graph_tables_to_pyg(
            tables,
            node_col_order=self.SCHEMA.node_col_order,
            edge_col_order=self.SCHEMA.edge_col_order,
            label_exprs=self.SCHEMA.label_exprs,
        )
        del num_graphs
        data.num_ids = num_ids
        return data, slices, num_raw


@dataclass(frozen=True)
class BaseGraphSource:
    """Catalog → shared vocab → train/val/test ``BaseGraphDataset``s.

    Subclass sets ``KIND`` (cache_key prefix) + ``DATASET_CLS``, and
    implements ``_scan_vocab`` returning sorted unique vocab values
    across the supplied subdirs.
    """

    KIND: ClassVar[str]
    DATASET_CLS: ClassVar[type[BaseGraphDataset]]

    name: str
    lake_root: str | None = None
    val_fraction: float = 0.2
    scaler_cfg: ScalerCfg = ZBenignScalerCfg()
    representation_cfg: GraphRepresentationCfg = field(default_factory=SnapshotRepresentationCfg)
    vocab_scope: Literal["train", "all"] = "train"

    def resolved_lake_root(self) -> str:
        if self.lake_root:
            return self.lake_root
        from graphids.paths import lake_root

        return lake_root()

    @property
    def cache_key(self) -> str:
        repr_digest = representation_digest(self.representation_cfg)
        window_size, stride = representation_window_defaults(self.representation_cfg)
        return (
            f"{self.KIND}|{self.resolved_lake_root()}|{self.name}"
            f"|w{window_size}|s{stride}"
            f"|v{self.val_fraction}"
            f"|sc:{scaler_kind(self.scaler_cfg)}|voc:{self.vocab_scope}"
            f"|repr:{representation_kind(self.representation_cfg)}:{repr_digest}"
        )

    def cache_root_path(self) -> Path:
        from graphids.paths import cache_dir

        lake = self.resolved_lake_root()
        repr_slug = (
            f"{representation_kind(self.representation_cfg)}_"
            f"{representation_digest(self.representation_cfg)}"
        )
        split_slug = f"val_{self.val_fraction:g}_gap_{split_embargo_width(self.representation_cfg)}"
        return cache_dir(lake, self.name) / f"{repr_slug}_voc_{self.vocab_scope}_{split_slug}"

    def cache_ready(self) -> bool:
        from graphids.paths import data_dir, load_catalog

        entry = load_catalog()[self.name]
        lake = self.resolved_lake_root()
        raw = data_dir(lake, self.name)
        root = self.cache_root_path()
        processed = root / "processed"
        train_dirs = [s for s in (entry.get("train_subdir"), entry.get("train_attack_subdir")) if s]
        if not train_dirs:
            return False
        present_test = [sd for sd in entry.get("test_subdirs", []) if (raw / sd).is_dir()]
        expected = [processed / "data_train.pt"] + [
            processed / f"data_test_{sd}.pt" for sd in present_test
        ]
        return (processed / ".complete").is_file() and all(path.is_file() for path in expected)

    def _scan_vocab(self, raw_dir: Path, source_dirs: list[str]) -> list[Any]:
        raise NotImplementedError(
            f"{type(self).__name__} must override _scan_vocab() to return "
            "sorted unique values of SCHEMA.vocab_column across all source_dirs."
        )

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
        """Optional hook for discovery or sidecar artifacts."""
        del root, raw, train_dirs, present_test, vocab, digest

    def build(self) -> DatasetState:
        from graphids.paths import data_dir, load_catalog

        entry = load_catalog()[self.name]
        lake = self.resolved_lake_root()
        root = self.cache_root_path()
        raw = data_dir(lake, self.name)

        train_dirs = [s for s in (entry.get("train_subdir"), entry.get("train_attack_subdir")) if s]
        if not train_dirs:
            raise ValueError(f"catalog entry {self.name!r} declares no train_subdir(s)")

        present_test = [sd for sd in entry.get("test_subdirs", []) if (raw / sd).is_dir()]
        scan_sources = list(train_dirs) + (present_test if self.vocab_scope == "all" else [])
        vocab = {tok: i + 1 for i, tok in enumerate(self._scan_vocab(raw, scan_sources))}
        digest = persist_vocab(vocab, Path(root) / "vocab.json")
        self._post_build_artifacts(
            root=Path(root),
            raw=raw,
            train_dirs=train_dirs,
            present_test=present_test,
            vocab=vocab,
            digest=digest,
        )

        common = dict(
            val_fraction=self.val_fraction,
            shared_vocab=vocab,
            shared_vocab_digest=digest,
            vocab_scope=self.vocab_scope,
            scaler_cfg=self.scaler_cfg,
            representation_cfg=self.representation_cfg,
        )
        train = self.DATASET_CLS(
            root=root, raw_dir=raw, split="train", source_dirs=train_dirs, **common,
        )
        val = self.DATASET_CLS(
            root=root, raw_dir=raw, split="val", source_dirs=train_dirs, **common,
        )
        tests = {
            sd: self.DATASET_CLS(
                root=root, raw_dir=raw, split="test", source_dirs=[sd], **common,
            )
            for sd in present_test
        }
        return DatasetState(train=train, val=val, test=tests)
