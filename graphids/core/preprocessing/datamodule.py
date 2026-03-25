"""LightningDataModule for CAN bus graph datasets.

Single DataModule for all 6 catalog datasets. Owns dataset construction,
train/val/test splits, and DataLoader creation via shared ``make_dataloader``.
"""

from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.config import cache_dir, data_dir
from graphids.config.constants import CATALOG_PATH, EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT, PREPROCESSING_DEFAULTS

from .datasets.can_bus import CANBusDataset

if TYPE_CHECKING:
    from collections.abc import Callable

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
        conv_type: str = "gatv2",
        heads: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train_ds: CANBusDataset | None = None
        self._val_ds: CANBusDataset | None = None
        self._test_datasets: dict[str, CANBusDataset] = {}

    @classmethod
    def from_cfg(cls, cfg) -> CANBusDataModule:
        """Construct from a resolved Config object."""
        from graphids.config.constants import STAGE_MODEL_MAP

        pre = cfg.preprocessing
        # Resolve conv_type/heads from the active model's sub-config.
        model_type = STAGE_MODEL_MAP.get(cfg.stage, cfg.model_type)
        model_sub = getattr(cfg, model_type, None)
        conv_type = getattr(model_sub, "conv_type", "gatv2") if model_sub else "gatv2"
        heads = getattr(model_sub, "heads", 4) if model_sub else 4
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
            conv_type=conv_type,
            heads=heads,
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
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from torch_geometric.loader import DynamicBatchSampler

        from graphids.core.models._training import compute_node_budget

        hp = self.hparams
        bs = max(8, hp["batch_size"])
        nw = hp["num_workers"] if "num_workers" in hp else 0

        common = dict(
            num_workers=nw,
            pin_memory=True,
            persistent_workers=nw > 0,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if hp["dynamic_batching"]:
            info = compute_node_budget(
                bs, hp, conv_type=hp.get("conv_type", "gatv2"), heads=hp.get("heads", 4),
            )
            num_steps = max(1, int(len(dataset) * info.mean_nodes / info.budget))
            sampler = DynamicBatchSampler(
                dataset, max_num=info.budget, mode="node", shuffle=shuffle,
                num_steps=num_steps, skip_too_big=True,
            )
            return PyGDataLoader(dataset, batch_sampler=sampler, **common)

        return PyGDataLoader(dataset, batch_size=bs, shuffle=shuffle, **common)


# ---------------------------------------------------------------------------
# Fusion data: cached state vectors from frozen VGAE + GAT
# ---------------------------------------------------------------------------


def cache_predictions(
    models: dict[str, torch.nn.Module],
    data,
    device: torch.device,
    max_samples: int = 150_000,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Run registered extractors over data, produce N-D state vectors for fusion.

    Uses a DataLoader for batched clone+transfer, then extracts per-graph
    features within each on-device batch (extractors are not batch-aware).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from graphids.core.models.registry import extractors as registry_extractors

    from ._graph_utils import get_batch_index

    active = [(name, ext) for name, ext in registry_extractors() if name in models]
    for model in models.values():
        model.eval()

    capped = data[:max_samples]
    loader = PyGDataLoader(capped, batch_size=batch_size, shuffle=False)

    states, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            for g in batch.to_data_list():
                batch_idx = get_batch_index(g, device)
                features = [ext.extract(models[name], g, batch_idx, device) for name, ext in active]
                states.append(torch.cat(features))
                labels.append(g.y[0] if g.y.dim() > 0 else g.y)

    return {"states": torch.stack(states), "labels": torch.tensor(labels)}


class FusionDataModule(pl.LightningDataModule):
    """Loads frozen VGAE+GAT, caches state vectors, serves DataLoaders.

    Wraps CANBusDataModule internally — callers never touch raw graph data.
    """

    def __init__(self, cfg, load_model_fn: Callable):
        super().__init__()
        self.cfg = cfg
        self._load_model = load_model_fn
        self._device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        is_rl = cfg.fusion.method in ("dqn", "bandit")
        self._batch_size = cfg.fusion.episode_sample_size if is_rl else cfg.dqn.batch_size
        self.train_cache: dict | None = None
        self.val_cache: dict | None = None

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_cache["states"]) / self._batch_size)

    def setup(self, stage=None):
        raw_dm = CANBusDataModule.from_cfg(self.cfg)
        raw_dm.setup("fit")
        raw_dm.populate_config(self.cfg)

        vgae = self._load_model(self.cfg, "vgae", "autoencoder", self._device)
        gat = self._load_model(self.cfg, "gat", self.cfg.gat_stage, self._device)
        models = {"vgae": vgae, "gat": gat}
        bs = self.cfg.evaluation.batch_size
        self.train_cache = cache_predictions(
            models, list(raw_dm.train_dataset), self._device, self.cfg.fusion.max_samples, batch_size=bs,
        )
        self.val_cache = cache_predictions(
            models, list(raw_dm.val_dataset), self._device, self.cfg.fusion.max_val_samples, batch_size=bs,
        )

        del vgae, gat, models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)
