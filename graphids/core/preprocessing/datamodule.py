"""LightningDataModule for CAN bus graph datasets.

Single DataModule for all 6 catalog datasets. Owns dataset construction,
train/val/test splits, and DataLoader creation.
"""

from __future__ import annotations

import gc
import os

import pytorch_lightning as pl
from graphids.log import get_logger
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.config import cache_dir, data_dir, load_catalog
from graphids.core.preprocessing.budget import node_budget

_log = get_logger(__name__)


from .datasets.can_bus import CANBusDataset
from .features import N_EDGE_FEATURES as EDGE_FEATURE_COUNT
from .features import N_NODE_FEATURES as NODE_FEATURE_COUNT


def _worker_init(worker_id: int) -> None:
    """Set file_system sharing strategy in spawn workers (not inherited from parent)."""
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")


def make_graph_loader(
    dataset, *, batch_sampler=None, batch_size=1, shuffle=False,
    num_workers: int = 0, pin_memory: bool = True,
    device: torch.device | None = None, **kwargs,
) -> DataLoader:
    """Thin wrapper around PyG DataLoader — sets spawn/persistent_workers defaults.

    Args:
        device: When set, wraps the loader with PyG's PrefetchLoader for async
            H2D transfer via CUDA streams. pin_memory is disabled on the inner
            loader (PrefetchLoader handles pinning internally).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    if device is not None:
        pin_memory = False  # PrefetchLoader pins internally

    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("worker_init_fn", _worker_init)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    if batch_sampler is not None:
        loader = PyGDataLoader(dataset, batch_sampler=batch_sampler, **common)
    else:
        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)

    if device is not None:
        from torch_geometric.loader import PrefetchLoader
        return PrefetchLoader(loader, device=device)
    return loader



log = get_logger(__name__)


def load_datasets(cfg) -> tuple[CANBusDataset, CANBusDataset, dict[str, CANBusDataset]]:
    """Load train/val/test datasets from cache. No DataModule needed.

    Returns (train_ds, val_ds, {name: test_ds}).
    """
    pre = cfg.preprocessing
    root = cache_dir(cfg.lake_root, cfg.dataset)
    raw = data_dir(cfg.lake_root, cfg.dataset)
    common = dict(
        window_size=pre.window_size, stride=pre.stride,
        val_fraction=1.0 - pre.train_val_split, seed=cfg.seed,
    )
    train_ds = CANBusDataset(root=root, raw_dir=raw, split="train", **common)
    train_ds._data_list = None
    val_ds = CANBusDataset(root=root, raw_dir=raw, split="val", **common)
    val_ds._data_list = None

    test_datasets = {}
    catalog = load_catalog()
    entry = catalog[cfg.dataset]
    for subdir in entry.get("test_subdirs", []):
        test_raw = raw / subdir
        if test_raw.exists():
            test_datasets[subdir] = CANBusDataset(root=root, raw_dir=test_raw, split="test", **common)

    return train_ds, val_ds, test_datasets


class CANBusDataModule(pl.LightningDataModule):
    """CAN bus graph data — one DataModule for all 6 catalog datasets.

    After ``setup()``, exposes ``train_dataset``, ``val_dataset``,
    ``test_datasets``, ``num_ids``, and ``in_channels`` as properties.
    """

    def __init__(
        self,
        dataset: str,
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        batch_size: int = 32,
        num_workers: int = 2,
        window_size: int = 100,
        stride: int = 100,
        val_fraction: float = 0.2,
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

    def setup(self, stage: str | None = None) -> None:
        import types
        hp = self.hparams
        cfg = types.SimpleNamespace(
            dataset=hp["dataset"], lake_root=hp["lake_root"], seed=hp["seed"],
            preprocessing=types.SimpleNamespace(
                window_size=hp["window_size"], stride=hp["stride"],
                train_val_split=1.0 - hp["val_fraction"],
            ),
        )
        self._train_ds, self._val_ds, self._test_datasets = load_datasets(cfg)

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
        n = int(ds._data.y.unique().numel())
        return n if n >= 2 else 2

    @property
    def edge_dim(self) -> int:
        """Edge feature dimensionality."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1] if len(ds) > 0 else EDGE_FEATURE_COUNT

    # -- DataLoaders ----------------------------------------------------------

    def train_dataloader(self):
        return self._build_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return [self._build_loader(ds, shuffle=False) for ds in self._test_datasets.values()]

    def _build_loader(self, dataset, shuffle: bool):
        from torch_geometric.loader import DynamicBatchSampler

        hp = self.hparams
        bs = max(8, hp["batch_size"])
        nw = hp["num_workers"] if "num_workers" in hp else 0

        # Async H2D via PrefetchLoader when GPU is available
        trainer = getattr(self, "trainer", None)
        device = None
        if trainer and torch.cuda.is_available():
            device = trainer.strategy.root_device

        if hp["dynamic_batching"]:
            model = trainer.lightning_module if trainer else None
            # conv_type/heads are model params, not data params — read from model
            model_hp = getattr(model, "hparams", {}) if model else {}
            result = node_budget(
                hp["dataset"], hp["lake_root"],
                conv_type=model_hp.get("conv_type", hp.get("conv_type", "gatv2")),
                heads=model_hp.get("heads", hp.get("heads", 4)),
                model=model, train_dataset=dataset,
                num_workers=nw,
            )
            import math as _math
            num_steps = max(1, _math.ceil(len(dataset) * result.mean_nodes / result.budget))
            sampler = DynamicBatchSampler(
                dataset, max_num=result.budget, mode="node", shuffle=shuffle,
                skip_too_big=True, num_steps=num_steps,
            )
            dataset._data_list = None  # clear bloat from sampler's __init__
            return make_graph_loader(dataset, batch_sampler=sampler, num_workers=nw, device=device)

        return make_graph_loader(dataset, batch_size=bs, shuffle=shuffle, num_workers=nw, device=device)


class FusionDataModule(pl.LightningDataModule):
    """Loads frozen VGAE+GAT, caches state vectors, serves DataLoaders.

    Wraps CANBusDataModule internally — callers never touch raw graph data.
    """

    def __init__(
        self,
        dataset: str = "",
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        vgae_ckpt_path: str = "",
        gat_ckpt_path: str = "",
        cached_states_dir: str = "",
        method: str = "bandit",
        batch_size: int = 128,
        episode_sample_size: int = 20000,
        max_samples: int = 150000,
        max_val_samples: int = 30000,
        eval_batch_size: int = 256,
        seed: int = 42,
        window_size: int = 100,
        stride: int = 100,
        val_fraction: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        is_rl = method in ("dqn", "bandit")
        self._batch_size = episode_sample_size if is_rl else batch_size
        self.train_cache: dict | None = None
        self.val_cache: dict | None = None

    @property
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.train_cache["states"]) / self._batch_size)

    @staticmethod
    def cache_predictions(
        models: dict[str, torch.nn.Module],
        data,
        device: torch.device,
        max_samples: int = 150_000,
        batch_size: int = 256,
    ) -> dict[str, torch.Tensor]:
        """Run registered extractors over data, produce N-D state vectors for fusion."""
        from graphids.core.models.fusion import extractors as registry_extractors

        active = [(name, ext) for name, ext in registry_extractors() if name in models]
        for model in models.values():
            model.eval()

        capped = data[:max_samples]
        loader = make_graph_loader(capped, batch_size=batch_size)

        states, labels = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                feats = [ext.extract(models[name], batch, device) for name, ext in active]
                states.append(torch.cat(feats, dim=1))  # [B, total_dim]
                labels.append(batch.y)

        return {"states": torch.cat(states), "labels": torch.cat(labels)}

    def setup(self, stage=None):
        if self.train_cache is not None:
            return

        hp = self.hparams

        # Fast path: load pre-extracted states from disk (no GPU needed)
        if hp.cached_states_dir:
            self._load_cached_states(hp.cached_states_dir)
            return

        # Slow path: load upstream models and extract on GPU
        import types
        from pathlib import Path

        from graphids.core.models._training import load_inner_model

        cfg_ns = types.SimpleNamespace(
            dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
            preprocessing=types.SimpleNamespace(
                window_size=hp.window_size, stride=hp.stride,
                train_val_split=1.0 - hp.val_fraction,
            ),
        )
        train_ds, val_ds, _ = load_datasets(cfg_ns)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not hp.vgae_ckpt_path:
            raise ValueError("vgae_ckpt_path is empty — upstream VGAE checkpoint not wired")
        if not hp.gat_ckpt_path:
            raise ValueError("gat_ckpt_path is empty — upstream GAT checkpoint not wired")
        vgae, _ = load_inner_model("vgae", Path(hp.vgae_ckpt_path), device)
        gat, _ = load_inner_model("gat", Path(hp.gat_ckpt_path), device)

        # Fusion pre-flight: warn if both models consume > 85% of VRAM
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
            usage_pct = allocated / total * 100
            if usage_pct > 85:
                log.warning("fusion_setup_vram_high",
                            allocated_mb=round(allocated / 1e6, 1),
                            total_mb=round(total / 1e6, 1),
                            pct=round(usage_pct, 1))

        models = {"vgae": vgae, "gat": gat}
        self.train_cache = self.cache_predictions(
            models, list(train_ds), device, hp.max_samples, batch_size=hp.eval_batch_size,
        )
        self.val_cache = self.cache_predictions(
            models, list(val_ds), device, hp.max_val_samples, batch_size=hp.eval_batch_size,
        )

        del vgae, gat, models
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_cached_states(self, cached_states_dir: str) -> None:
        """Load pre-extracted fusion states from disk. No GPU needed."""
        from pathlib import Path
        from graphids.commands.extract_fusion_states import (
            FUSION_STATES_DIR, TRAIN_FILENAME, VAL_FILENAME,
        )

        states_dir = Path(cached_states_dir)
        # Support both direct dir and parent dir containing fusion_states/
        if not (states_dir / TRAIN_FILENAME).exists():
            states_dir = states_dir / FUSION_STATES_DIR
        train_path = states_dir / TRAIN_FILENAME
        val_path = states_dir / VAL_FILENAME
        if not train_path.exists():
            raise FileNotFoundError(f"Cached train states not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Cached val states not found: {val_path}")

        self.train_cache = torch.load(train_path, map_location="cpu", weights_only=True)
        self.val_cache = torch.load(val_path, map_location="cpu", weights_only=True)
        log.info("loaded_cached_states",
                 dir=str(states_dir),
                 train_shape=list(self.train_cache["states"].shape),
                 val_shape=list(self.val_cache["states"].shape))

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)

    def test_dataloader(self):
        return self.val_dataloader()
