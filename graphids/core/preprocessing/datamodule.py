"""LightningDataModule for CAN bus graph datasets.

Single DataModule for all 6 catalog datasets. Owns dataset construction,
train/val/test splits, and DataLoader creation.
"""

from __future__ import annotations

import gc
import json
import math
import os

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.config import cache_dir, data_dir
from graphids.config import CATALOG_PATH

_log = structlog.get_logger()

# --- Dynamic batching constants -------------------------------------------
# Conv types with O(N²) global attention (full attention matrix).
_QUADRATIC_CONV_TYPES = frozenset({"gps"})
# Fallback activation cost per node when no model is available for probing.
_BYTES_PER_NODE = 32_768
# Reserve 15% of free VRAM for allocator fragmentation + edge-density variance.
_SAFETY_MARGIN = 0.85
# Forward-only probe captures activations; multiply by this to account for
# gradient memory during backward (gradients ≈ activations in size).
_GRAD_MULTIPLIER = 2


def _probe_bytes_per_node(model, dataset, n_target: int = 2000, step_fn=None) -> int:
    """Run a representative step, measure peak VRAM, return estimated bytes/node.

    When *step_fn* is provided (e.g. ``model._step``), the probe measures the
    full training-step footprint — including KD teacher inference and any other
    computation in ``_step``.  Falls back to ``model.forward`` when *step_fn*
    is ``None``.

    Runs under torch.no_grad() and multiplies by _GRAD_MULTIPLIER to account
    for gradient memory during real training.
    """
    from torch_geometric.data import Batch

    graphs, n = [], 0
    for g in dataset:
        graphs.append(g)
        n += g.num_nodes
        if n >= n_target:
            break

    batch = Batch.from_data_list(graphs).to(model.device)
    actual_nodes = batch.num_nodes

    torch.cuda.reset_peak_memory_stats(model.device)
    before = torch.cuda.memory_allocated(model.device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        (step_fn or model)(batch)
    model.train(was_training)

    peak = torch.cuda.max_memory_allocated(model.device)
    del batch
    torch.cuda.empty_cache()

    fwd_per_node = max(1, int((peak - before) / actual_nodes))
    bpn = fwd_per_node * _GRAD_MULTIPLIER
    _log.info("probe_bytes_per_node", bytes_per_node=bpn, probe_nodes=actual_nodes,
              fwd_per_node=fwd_per_node, delta_mb=round((peak - before) / 1e6, 1),
              method="step_fn" if step_fn else "forward")
    return bpn


def vram_node_budget(
    dataset: str, lake_root: str, *, conv_type: str = "gatv2", heads: int = 4,
    model=None, train_dataset=None,
) -> tuple[int, float]:
    """Compute node budget from available VRAM and dataset graph stats.

    If *model* and *train_dataset* are provided and CUDA is available, probes
    actual per-node activation memory.  When the model exposes a ``_step``
    method (all graph LightningModules do), the probe runs the full training
    step — capturing KD teacher inference, auxiliary losses, etc. — instead of
    just ``forward()``.

    Returns (node_budget, mean_nodes).
    """
    metadata_path = cache_dir(lake_root, dataset) / "cache_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"cache_metadata.json not found at {metadata_path}. "
            "Run preprocessing first."
        )
    stats = json.loads(metadata_path.read_text())["graph_stats"]["node_count"]
    mean_nodes = stats["mean"]

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
    else:
        free = 12 * 1024**3  # CPU fallback for testing

    if conv_type in _QUADRATIC_CONV_TYPES:
        budget = int(math.sqrt(free / (heads * 3 * 2)))
        _log.info("vram_node_budget", conv_type=conv_type, budget=budget,
                  free_vram_gb=round(free / 1e9, 2), mean_nodes=mean_nodes,
                  method="quadratic")
        return budget, mean_nodes

    if model is not None and train_dataset is not None and torch.cuda.is_available():
        # Prefer _step over forward — captures full training footprint (KD, etc.)
        step_fn = getattr(model, "_step", None)
        bytes_per_node = _probe_bytes_per_node(model, train_dataset, step_fn=step_fn)
    else:
        bytes_per_node = _BYTES_PER_NODE

    budget = int(free * _SAFETY_MARGIN / bytes_per_node)

    _log.info("vram_node_budget", conv_type=conv_type, budget=budget,
              free_vram_gb=round(free / 1e9, 2), mean_nodes=mean_nodes,
              bytes_per_node=bytes_per_node,
              method="probe" if model is not None else "fallback")
    return budget, mean_nodes

from .features import N_EDGE_FEATURES as EDGE_FEATURE_COUNT, N_NODE_FEATURES as NODE_FEATURE_COUNT

from .datasets.can_bus import CANBusDataset


def _worker_init(worker_id: int) -> None:
    """Set file_system sharing strategy in spawn workers (not inherited from parent)."""
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")


def make_graph_loader(
    dataset, *, batch_sampler=None, batch_size=1, shuffle=False,
    num_workers: int = 0, pin_memory: bool = True, **kwargs,
) -> DataLoader:
    """Thin wrapper around PyG DataLoader — sets spawn/persistent_workers defaults."""
    from torch_geometric.loader import DataLoader as PyGDataLoader

    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("worker_init_fn", _worker_init)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    if batch_sampler is not None:
        return PyGDataLoader(dataset, batch_sampler=batch_sampler, **common)
    return PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)



log = structlog.get_logger()


def _load_catalog() -> dict:
    import yaml

    return yaml.safe_load(CATALOG_PATH.read_text())


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
    catalog = _load_catalog()
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

        if hp["dynamic_batching"]:
            trainer = getattr(self, "trainer", None)
            model = trainer.lightning_module if trainer else None
            budget, mean_nodes = vram_node_budget(
                hp["dataset"], hp["lake_root"],
                conv_type=hp.get("conv_type", "gatv2"),
                heads=hp.get("heads", 4),
                model=model, train_dataset=dataset,
            )
            num_steps = max(1, int(len(dataset) * mean_nodes / budget))
            sampler = DynamicBatchSampler(
                dataset, max_num=budget, mode="node", shuffle=shuffle,
                skip_too_big=True, num_steps=num_steps,
            )
            dataset._data_list = None  # clear bloat from sampler's __init__
            return make_graph_loader(dataset, batch_sampler=sampler, num_workers=nw)

        return make_graph_loader(dataset, batch_size=bs, shuffle=shuffle, num_workers=nw)


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
        from graphids.core.models.registry import extractors as registry_extractors

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
        import types
        from pathlib import Path
        from graphids.core.models._training import load_inner_model

        hp = self.hparams
        cfg_ns = types.SimpleNamespace(
            dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
            preprocessing=types.SimpleNamespace(
                window_size=hp.window_size, stride=hp.stride,
                train_val_split=1.0 - hp.val_fraction,
            ),
        )
        train_ds, val_ds, _ = load_datasets(cfg_ns)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        vgae, _ = load_inner_model("vgae", Path(hp.vgae_ckpt_path), device)
        gat, _ = load_inner_model("gat", Path(hp.gat_ckpt_path), device)
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

    def train_dataloader(self):
        ds = TensorDataset(self.train_cache["states"], self.train_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size, shuffle=True)

    def val_dataloader(self):
        ds = TensorDataset(self.val_cache["states"], self.val_cache["labels"])
        return DataLoader(ds, batch_size=self._batch_size)
