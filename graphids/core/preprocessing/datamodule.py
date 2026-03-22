"""LightningDataModule for CAN bus graph datasets.

Single DataModule for all 6 catalog datasets. Owns dataset construction,
train/val/test splits, and DataLoader creation via shared ``make_dataloader``.
"""

from __future__ import annotations

import structlog

import pytorch_lightning as pl

from graphids.config import cache_dir, data_dir
from graphids.config.constants import CATALOG_PATH, EDGE_FEATURE_COUNT, PREPROCESSING_DEFAULTS

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
        safety_factor: float = 1.0,
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
            safety_factor=cfg.training.safety_factor,
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
        return ds[0].x.shape[1] if len(ds) > 0 else 31

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
        # Lazy import to avoid circular: datamodule (core) -> data_loading (pipeline)
        from graphids.pipeline.stages.data_loading import compute_node_budget, make_dataloader

        hp = self.hparams
        bs = max(8, int(hp["batch_size"] * hp["safety_factor"]))
        max_nodes = compute_node_budget(bs, hp) if hp["dynamic_batching"] else None
        return make_dataloader(dataset, hp, bs, shuffle=shuffle, max_num_nodes=max_nodes)
