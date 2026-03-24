"""LightningDataModule for CAN bus graph datasets.

Single DataModule for all 6 catalog datasets. Owns dataset construction,
train/val/test splits, and DataLoader creation via shared ``make_dataloader``.
"""

from __future__ import annotations

import structlog

import pytorch_lightning as pl

from graphids.config import cache_dir, data_dir
from graphids.config.constants import CATALOG_PATH, EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT, PREPROCESSING_DEFAULTS

from .datasets.can_bus import CANBusDataset

log = structlog.get_logger()


def _load_catalog() -> dict:
    import yaml

    return yaml.safe_load(CATALOG_PATH.read_text())


class CANBusDataModule(pl.LightningDataModule):
    """CAN bus graph data — one DataModule for all 6 catalog datasets.

    After ``setup()``, exposes ``train_dataset``, ``val_dataset``,
    ``test_datasets``, ``num_ids``, and ``in_channels`` as properties.
    """

    def __init__(
        self,
        dataset: str,
        lake_root: str,
        batch_size: int = 32,
        num_workers: int = 2,
        window_size: int = PREPROCESSING_DEFAULTS["window_size"],
        stride: int = PREPROCESSING_DEFAULTS["stride"],
        val_fraction: float = 1.0 - PREPROCESSING_DEFAULTS["train_val_split"],
        seed: int = 42,
        dynamic_batching: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train_ds: CANBusDataset | None = None
        self._val_ds: CANBusDataset | None = None
        self._test_datasets: dict[str, CANBusDataset] = {}

    @classmethod
    def from_cfg(cls, cfg) -> CANBusDataModule:
        """Construct from a resolved Config object."""
        pre = cfg.preprocessing
        return cls(
            dataset=cfg.dataset,
            lake_root=cfg.lake_root,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.num_workers,
            seed=cfg.seed,
            dynamic_batching=cfg.training.dynamic_batching,
            window_size=pre.window_size,
            stride=pre.stride,
            val_fraction=1.0 - pre.train_val_split,
        )

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        root = cache_dir(hp["lake_root"], hp["dataset"])
        raw = data_dir(hp["lake_root"], hp["dataset"])
        common = dict(
            window_size=hp["window_size"],
            stride=hp["stride"],
            val_fraction=hp["val_fraction"],
            seed=hp["seed"],
        )
        if stage in ("fit", None):
            self._train_ds = CANBusDataset(root=root, raw_dir=raw, split="train", **common)
            self._val_ds = CANBusDataset(root=root, raw_dir=raw, split="val", **common)
        if stage in ("test", None):
            catalog = _load_catalog()
            entry = catalog[hp["dataset"]]
            for subdir in entry.get("test_subdirs", []):
                test_raw = raw / subdir
                if not test_raw.exists():
                    log.warning("test_subdir_missing", subdir=subdir, raw_dir=str(raw))
                    continue
                self._test_datasets[subdir] = CANBusDataset(
                    root=root, raw_dir=test_raw, split="test", **common,
                )

    # -- Properties (available after setup) -----------------------------------

    @property
    def train_dataset(self) -> CANBusDataset:
        assert self._train_ds is not None, "call setup('fit') first"
        return self._train_ds

    @property
    def val_dataset(self) -> CANBusDataset:
        assert self._val_ds is not None, "call setup('fit') first"
        return self._val_ds

    @property
    def test_datasets(self) -> dict[str, CANBusDataset]:
        return self._test_datasets

    @property
    def num_ids(self) -> int:
        """Global CAN arbitration-ID vocabulary size (embedding table size)."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds.num_arb_ids

    @property
    def in_channels(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].x.shape[1] if len(ds) > 0 else NODE_FEATURE_COUNT

    @property
    def num_classes(self) -> int:
        """Number of unique target classes across the dataset (fallback to 2)."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        if len(ds) == 0:
            return 2
        import torch
        labels = torch.cat([g.y.view(-1) for g in ds])
        n = int(labels.unique().numel())
        return n if n >= 2 else 2

    @property
    def edge_dim(self) -> int:
        """Edge feature dimensionality."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1] if len(ds) > 0 else EDGE_FEATURE_COUNT

    def populate_config(self, cfg) -> None:
        """Write data-derived dimensions (num_ids, in_channels, num_classes) into cfg.

        Must be called after setup(). Eliminates manual threading of these
        values through every stage function and model constructor.
        """
        from omegaconf import OmegaConf, open_dict

        with open_dict(cfg):
            cfg.num_ids = self.num_ids
            cfg.in_channels = self.in_channels
            cfg.num_classes = self.num_classes
            cfg.vgae.edge_dim = self.edge_dim
            cfg.gat.edge_dim = self.edge_dim

    # -- DataLoaders ----------------------------------------------------------

    def train_dataloader(self):
        return self._build_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return [self._build_loader(ds, shuffle=False) for ds in self._test_datasets.values()]

    def _build_loader(self, dataset, shuffle: bool):
        import json

        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch_geometric.loader import DynamicBatchSampler

        hp = self.hparams
        bs = max(8, hp["batch_size"])
        nw = hp["num_workers"] if "num_workers" in hp else 0

        max_nodes, mean_nodes = None, None
        if hp["dynamic_batching"]:
            metadata_path = cache_dir(hp["lake_root"], hp["dataset"]) / "cache_metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"cache_metadata.json not found at {metadata_path}. "
                    "Rebuild caches with: python -m graphids stage=preprocess dataset=..."
                )
            stats = json.loads(metadata_path.read_text())["graph_stats"]["node_count"]
            max_nodes = int(bs * stats["p95"])
            mean_nodes = stats["mean"]

        common = dict(
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if max_nodes is not None:
            num_steps = max(1, int(len(dataset) * mean_nodes / max_nodes))
            sampler = DynamicBatchSampler(
                dataset, max_num=max_nodes, mode="node", shuffle=shuffle,
                num_steps=num_steps, skip_too_big=True,
            )
            return PyGDataLoader(dataset, batch_sampler=sampler, **common)

        return PyGDataLoader(dataset, batch_size=bs, shuffle=shuffle, **common)
