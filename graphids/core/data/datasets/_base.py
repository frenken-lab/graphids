"""Base dataset and source primitives for graph preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

import polars as pl
import torch
from filelock import FileLock
from structlog import get_logger
from torch_geometric.data import Data, InMemoryDataset

from graphids._fs import atomic_save
from graphids.core.data.preprocessing import scaler as scaler_mod
from graphids.core.data.preprocessing.edge_policy import EdgePolicy
from graphids.core.data.preprocessing.graph_ops import GraphTransform
from graphids.core.data.preprocessing.metadata import (
    load_metadata,
    merge_split_into_metadata,
)
from graphids.core.data.preprocessing.pipeline import (
    GraphPipeline,
    GraphTables,
)
from graphids.core.data.preprocessing.pipeline import (
    build_tables as build_pipeline_tables,
)
from graphids.core.data.preprocessing.pipeline import (
    run as run_pipeline,
)
from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    representation_digest,
    representation_kind,
    representation_payload,
    representation_segment,
    representation_window_defaults,
)
from graphids.core.data.preprocessing.scaler import (
    ScalerCfg,
    ZBenignScalerCfg,
    scaler_kind,
)
from graphids.core.data.preprocessing.splits import (
    SPLIT_POLICY,
    audit_split_plan,
    build_blocked_split_plan,
    split_embargo_width,
    split_policy_digest,
)
from graphids.core.data.preprocessing.vocab import persist_vocab
from graphids.core.data.state import DatasetState
from graphids.paths import PREPROCESSING_VERSION

log = get_logger(__name__)

_DEFAULT_SCALER_CFG = ZBenignScalerCfg()
_DEFAULT_REPRESENTATION_CFG = SnapshotRepresentationCfg()


def _stats(t: torch.Tensor) -> dict[str, float | int]:
    if t.numel() == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "min": int(t.min()),
        "max": int(t.max()),
        "mean": float(t.mean()),
        "p95": float(t.quantile(0.95)),
        "p99": float(t.quantile(0.99)),
    }


def _balance(t: torch.Tensor) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in t.tolist():
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return out


@dataclass(frozen=True)
class GraphSchema:
    node_stat_exprs: list[pl.Expr]
    edge_stat_exprs: list[pl.Expr]
    node_col_order: list[str]
    edge_col_order: tuple[str, ...]
    label_exprs: list[pl.Expr]
    edge_base_cols: list[str]
    vocab_column: str
    attack_type_codes: dict[str, int] | None = None
    attack_type_names: dict[int, str] | None = None
    edge_policy: EdgePolicy | None = None
    graph_transforms: tuple[GraphTransform, ...] | None = None


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
        seed: int = 42,
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
        self.seed = seed
        self._shared_vocab = shared_vocab
        self._shared_vocab_digest = shared_vocab_digest
        self.vocab_scope = vocab_scope
        self.scaler_cfg = scaler_cfg
        self.scaler_strategy = scaler_kind(scaler_cfg)
        self.representation_cfg = representation_cfg
        self.representation_kind = representation_kind(representation_cfg)
        self._split_plan = None
        self.window_size, self.stride = (
            self._resolved_window_size_stride(representation_cfg)
        )
        super().__init__(str(root), transform, pre_transform)
        self.load(self.processed_paths[0])
        self.num_ids = int(load_metadata(Path(self.root))["num_arb_ids"])
        if self.split in ("train", "val"):
            self._split_plan = build_blocked_split_plan(
                self._data,
                self.slices,
                self.representation_cfg,
                val_fraction=self.val_fraction,
                seed=self.seed,
            )
            idx = self._split_plan.val_idx if self.split == "val" else self._split_plan.train_idx
            self._indices = idx.tolist()

    # ── must override ──────────────────────────────────────────────────
    def _read_raw(self) -> pl.DataFrame:
        raise NotImplementedError(
            f"{type(self).__name__}._read_raw must return a long-format pl.DataFrame"
        )

    # ── default ────────────────────────────────────────────────────────
    def _infer_attack_type(self, csv: Path) -> int:
        codes = self.SCHEMA.attack_type_codes or {}
        s = csv.stem.lower() + " " + csv.parent.name.lower()
        for kw, code in codes.items():
            if kw in s:
                return code
        return 0

    @staticmethod
    def _resolved_window_size_stride(representation_cfg: GraphRepresentationCfg) -> tuple[int, int]:
        return representation_window_defaults(representation_cfg)

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

    def process(self) -> None:
        with FileLock(str(Path(self.processed_dir) / ".lock")):
            marker = Path(self.processed_dir) / ".complete"
            tensor_path = Path(self.processed_paths[0])
            if tensor_path.exists() and marker.exists():
                return

            data, slices, num_arb_ids, num_graphs, num_raw = self._build_graphs()

            scaler_path = Path(self.processed_dir) / "feature_scaler.pt"
            split_plan = (
                build_blocked_split_plan(
                    data,
                    slices,
                    self.representation_cfg,
                    val_fraction=self.val_fraction,
                    seed=self.seed,
                )
                if self.split in ("train", "val")
                else None
            )

            if self.split == "train":
                if split_plan is None:
                    raise RuntimeError("train split missing SplitPlan")
                train_idx = split_plan.train_idx
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

            invariants = {
                "preprocessing_version": PREPROCESSING_VERSION,
                "window_size": self.window_size,
                "stride": self.stride,
                "val_fraction": self.val_fraction,
                "seed": self.seed,
                "vocab_digest": self._shared_vocab_digest,
                "scaler_strategy": self.scaler_strategy,
                "representation_kind": self.representation_kind,
                "representation_digest": representation_digest(self.representation_cfg),
                "representation_cfg": representation_payload(self.representation_cfg),
                "vocab_scope": self.vocab_scope,
            }
            if split_plan is None:
                split_meta = {
                    "split_policy": SPLIT_POLICY,
                    "split_unit": "dense_base_window",
                    "split_embargo": split_embargo_width(self.representation_cfg),
                    "split_plan_digest": split_policy_digest(
                        self.representation_cfg,
                        val_fraction=self.val_fraction,
                        seed=self.seed,
                    ),
                }
            else:
                split_meta = split_plan.metadata()
            invariants.update(
                {
                    "split_policy": split_meta["split_policy"],
                    "split_unit": split_meta["split_unit"],
                    "split_embargo": split_meta["split_embargo"],
                    "split_plan_digest": split_meta["split_plan_digest"],
                }
            )
            common = dict(
                invariants=invariants,
                dataset_name=Path(self.root).name,
                num_arb_ids=num_arb_ids,
            )
            bytes_on_disk = tensor_path.stat().st_size

            if self.split == "train":
                if split_plan is None:
                    raise RuntimeError("train split missing SplitPlan")
                train_idx, val_idx = split_plan.train_idx, split_plan.val_idx
                split_audit = audit_split_plan(split_plan)
                merge_split_into_metadata(
                    Path(self.root),
                    "train",
                    {
                        **self._split_entry(data, slices, train_idx, num_raw, bytes_on_disk),
                        **split_plan.metadata(),
                        "split_audit": split_audit,
                    },
                    **common,
                )
                merge_split_into_metadata(
                    Path(self.root),
                    "val",
                    {
                        **self._split_entry(data, slices, val_idx, num_raw, bytes_on_disk),
                        "derived_from": "train",
                        "val_fraction_seed": [self.val_fraction, self.seed],
                        "num_raw_samples": 0,
                        "bytes_on_disk": 0,
                        **split_plan.metadata(),
                        "split_audit": split_audit,
                    },
                    **common,
                )
            else:
                merge_split_into_metadata(
                    Path(self.root),
                    self.cache_split_name,
                    self._split_entry(data, slices, None, num_raw, bytes_on_disk),
                    **common,
                )
            marker.write_text("ok")

    def _split_entry(
        self,
        data: Data,
        slices: dict,
        indices: torch.Tensor | None,
        num_raw: int,
        bytes_on_disk: int,
    ) -> dict:
        node_diffs = (slices["x"][1:] - slices["x"][:-1]).float()
        edge_diffs = (slices["edge_index"][1:] - slices["edge_index"][:-1]).float()
        attack = data.attack_type
        if indices is not None:
            idx = torch.as_tensor(indices, dtype=torch.long)
            node_diffs = node_diffs.index_select(0, idx)
            edge_diffs = edge_diffs.index_select(0, idx)
            attack = attack.index_select(0, idx)

        names = self.SCHEMA.attack_type_names or {0: "benign"}
        balance: dict[str, int] = {}
        for t in attack.tolist():
            name = names.get(int(t), f"unknown_{int(t)}")
            balance[name] = balance.get(name, 0) + 1

        entry: dict = {
            "num_graphs": int(node_diffs.numel()),
            "graph_stats": {"node_count": _stats(node_diffs), "edge_count": _stats(edge_diffs)},
            "label_balance": _balance(data.y.index_select(0, idx) if indices is not None else data.y),
            "attack_balance": balance,
            "num_raw_samples": int(num_raw),
            "bytes_on_disk": int(bytes_on_disk),
        }
        if self.source_dirs is not None:
            entry["source_dirs"] = list(self.source_dirs)
        return entry

    def _build_graphs(self) -> tuple[Data, dict, int, int, int]:
        df = self._read_raw()
        log.info("raw_loaded", rows=len(df))
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

    def build_graph_tables(self) -> GraphTables:
        """Return staged graph tables for exploratory analysis before tensor packing."""
        return build_pipeline_tables(
            self._graph_pipeline(),
            self._with_vocab(self._read_raw()),
        )

    def _build_graphs_from_df(self, df: pl.DataFrame, num_ids: int) -> tuple[Data, dict, int, int, int]:
        data, slices, num_graphs, num_raw = run_pipeline(self._graph_pipeline(), df)
        del df
        return data, slices, num_ids, num_graphs, num_raw

    def _graph_pipeline(self) -> GraphPipeline:
        segment_cfg = representation_segment(self.representation_cfg)
        return GraphPipeline(
            node_stat_exprs=self.SCHEMA.node_stat_exprs,
            edge_stat_exprs=self.SCHEMA.edge_stat_exprs,
            node_col_order=self.SCHEMA.node_col_order,
            edge_col_order=self.SCHEMA.edge_col_order,
            label_exprs=self.SCHEMA.label_exprs,
            edge_base_cols=self.SCHEMA.edge_base_cols,
            edge_policy=self.SCHEMA.edge_policy,
            graph_transforms=list(self.SCHEMA.graph_transforms)
            if self.SCHEMA.graph_transforms is not None
            else None,
            representation_cfg=self.representation_cfg,
            segment_cfg=segment_cfg,
        )

    def _with_vocab(self, df: pl.DataFrame) -> pl.DataFrame:
        if self._shared_vocab is None:
            raise ValueError(
                f"{type(self).__name__} needs shared_vocab for split={self.split!r}; "
                "build via the source's build() so vocab is scanned across splits"
            )
        return df.with_columns(
            pl.col(self.SCHEMA.vocab_column)
            .replace_strict(self._shared_vocab, default=0)
            .cast(pl.Int64)
            .alias("node_id")
        )


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
    seed: int = 42
    scaler_cfg: ScalerCfg = ZBenignScalerCfg()
    representation_cfg: GraphRepresentationCfg = field(default_factory=SnapshotRepresentationCfg)
    vocab_scope: Literal["train", "all"] = "train"

    @property
    def window_size(self) -> int:
        return representation_window_defaults(self.representation_cfg)[0]

    @property
    def stride(self) -> int:
        return representation_window_defaults(self.representation_cfg)[1]

    def resolved_lake_root(self) -> str:
        if self.lake_root:
            return self.lake_root
        from graphids.paths import lake_root

        return lake_root()

    @property
    def cache_key(self) -> str:
        repr_digest = representation_digest(self.representation_cfg)
        return (
            f"{self.KIND}|{self.resolved_lake_root()}|{self.name}"
            f"|w{self.window_size}|s{self.stride}"
            f"|v{self.val_fraction}|seed{self.seed}"
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
        split_slug = split_policy_digest(
            self.representation_cfg,
            val_fraction=self.val_fraction,
            seed=self.seed,
        )
        return cache_dir(lake, self.name) / f"{repr_slug}_voc_{self.vocab_scope}_split_{split_slug}"

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
            seed=self.seed,
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
