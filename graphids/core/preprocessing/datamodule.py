"""LightningDataModules for graph datasets.

- ``GraphDataModule``: dataset-agnostic base. Subclasses set ``dataset_cls``.
- ``CANBusDataModule``: 3-line subclass binding ``CANBusDataset``.
- ``CurriculumDataModule``: curriculum-ordered sampling over CAN data.
- ``FusionDataModule``: frozen VGAE+GAT → cached state tensors.
"""

from __future__ import annotations

import gc
import math
import os
from pathlib import Path
from typing import ClassVar

import pytorch_lightning as pl
import torch
from graphids.log import get_logger
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import InMemoryDataset

from graphids.config import cache_dir, data_dir, load_catalog
from graphids.core.preprocessing.budget import (
    calibrate_at_budget,
    compute_resource_profile,
    node_budget,
)
from graphids.core.preprocessing.datasets.can_bus import (
    N_EDGE_FEATURES as EDGE_FEATURE_COUNT,
)
from graphids.core.preprocessing.datasets.can_bus import (
    N_NODE_FEATURES as NODE_FEATURE_COUNT,
)
from graphids.core.preprocessing.datasets.can_bus import CANBusDataset
from graphids.core.preprocessing.sampler import (
    CurriculumSampler,
    NodeBudgetBatchSampler,
    make_graph_loader,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared dataset loading
# ---------------------------------------------------------------------------


def load_datasets(
    cfg, dataset_cls: type[InMemoryDataset],
) -> tuple[InMemoryDataset, InMemoryDataset, dict[str, InMemoryDataset]]:
    """Load train/val/test datasets from cache using the given dataset class.

    Returns (train_ds, val_ds, {name: test_ds}).
    """
    pre = cfg.preprocessing
    root = cache_dir(cfg.lake_root, cfg.dataset)
    raw = data_dir(cfg.lake_root, cfg.dataset)
    common = dict(
        window_size=pre.window_size, stride=pre.stride,
        val_fraction=1.0 - pre.train_val_split, seed=cfg.seed,
    )
    train_ds = dataset_cls(root=root, raw_dir=raw, split="train", **common)
    train_ds._data_list = None
    val_ds = dataset_cls(root=root, raw_dir=raw, split="val", **common)
    val_ds._data_list = None

    test_datasets = {}
    catalog = load_catalog()
    entry = catalog[cfg.dataset]
    for subdir in entry.get("test_subdirs", []):
        test_raw = raw / subdir
        if test_raw.exists():
            test_datasets[subdir] = dataset_cls(root=root, raw_dir=test_raw, split="test", **common)

    return train_ds, val_ds, test_datasets


# ---------------------------------------------------------------------------
# GraphDataModule — dataset-agnostic base
# ---------------------------------------------------------------------------


class GraphDataModule(pl.LightningDataModule):
    """Dataset-agnostic graph DataModule.

    Subclasses set ``dataset_cls`` to a concrete ``InMemoryDataset`` subclass.
    All batching, VRAM sizing, and loader construction logic is shared.
    """

    dataset_cls: ClassVar[type[InMemoryDataset]]

    def __init__(
        self,
        dataset: str,
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        batch_size: int = 32,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
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
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}

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
        self._train_ds, self._val_ds, self._test_datasets = load_datasets(
            cfg, self.dataset_cls,
        )

    # -- Properties (available after setup) -----------------------------------

    @property
    def train_dataset(self) -> InMemoryDataset:
        assert self._train_ds is not None, "call setup('fit') first"
        return self._train_ds

    @property
    def val_dataset(self) -> InMemoryDataset:
        assert self._val_ds is not None, "call setup('fit') first"
        return self._val_ds

    @property
    def test_datasets(self) -> dict[str, InMemoryDataset]:
        return self._test_datasets

    @property
    def num_ids(self) -> int:
        """Global arbitration/node-ID vocabulary size (embedding table size)."""
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
        hp = self.hparams
        bs = max(8, hp["batch_size"])
        nw = hp.get("num_workers")  # None = auto-size from sizing chain
        pf = hp.get("prefetch_factor", 2)

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
            )

            # Auto-size workers: calibrate at the actual operating batch size
            if nw is None:
                slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
                max_cpus = int(slurm_cpus) if slurm_cpus else os.cpu_count()
                bwd_mult = result.backward_multiplier or 2.0
                t_c, t_g, _n_graphs = calibrate_at_budget(
                    model, dataset, result.budget,
                    backward_multiplier=bwd_mult,
                )
                profile = compute_resource_profile(
                    result,
                    t_collation_s=t_c if t_c > 0 else None,
                    t_gpu_s=t_g if t_g > 0 else None,
                    max_cpus=max_cpus,
                )
                if profile is not None:
                    nw = profile.workers
                    pf = profile.prefetch_factor
                else:
                    nw = 2  # fallback when probe unavailable

            num_steps = max(1, math.ceil(len(dataset) * result.mean_nodes / result.budget))
            # Read per-graph sizes from the cache's slice offsets (zero I/O).
            # Replaces PyG's DynamicBatchSampler, which walks dataset[i].num_nodes
            # per graph per epoch.
            sizes = dataset.num_nodes_per_graph
            sampler = NodeBudgetBatchSampler(
                sizes, max_num=result.budget, shuffle=shuffle,
                skip_too_big=True, num_steps=num_steps,
            )
            return make_graph_loader(
                dataset, batch_sampler=sampler, num_workers=nw, device=device,
                prefetch_factor=pf if nw > 0 else None,
            )

        if nw is None:
            nw = 2
        return make_graph_loader(
            dataset, batch_size=bs, shuffle=shuffle, num_workers=nw, device=device,
            prefetch_factor=pf if nw > 0 else None,
        )


# ---------------------------------------------------------------------------
# CANBusDataModule — thin subclass binding CANBusDataset
# ---------------------------------------------------------------------------


class CANBusDataModule(GraphDataModule):
    """CAN bus graph data — one DataModule for all 6 catalog datasets."""

    dataset_cls = CANBusDataset


# ---------------------------------------------------------------------------
# CurriculumDataModule — difficulty-ordered sampling over CAN data
# ---------------------------------------------------------------------------


class CurriculumDataModule(CANBusDataModule):
    """Curriculum learning with persistent workers.

    Subclasses CANBusDataModule for data loading + properties (num_ids,
    in_channels, num_classes). Adds VGAE difficulty scoring and
    curriculum-ordered batching via CurriculumSampler.
    """

    def __init__(
        self,
        dataset: str = "",
        lake_root: str = os.environ.get("KD_GAT_LAKE_ROOT"),
        vgae_ckpt_path: str = "",
        batch_size: int = 8192,
        num_workers: int | None = None,
        prefetch_factor: int = 2,
        window_size: int = 100,
        stride: int = 100,
        val_fraction: float = 0.2,
        seed: int = 42,
        dynamic_batching: bool = True,
        conv_type: str = "gatv2",
        heads: int = 4,
        canid_weight: float = 0.1,
        curriculum_start_ratio: float = 1.0,
        curriculum_end_ratio: float = 10.0,
        difficulty_percentile: float = 75.0,
        max_epochs: int = 300,
    ):
        super().__init__(
            dataset=dataset, lake_root=lake_root, batch_size=batch_size,
            num_workers=num_workers, prefetch_factor=prefetch_factor,
            window_size=window_size, stride=stride,
            val_fraction=val_fraction, seed=seed, dynamic_batching=dynamic_batching,
            conv_type=conv_type, heads=heads,
        )
        self.save_hyperparameters()
        self._batch_sampler = None
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage=None):
        if self._train_loader is not None:
            return
        from graphids.core.models._training import load_inner_model

        hp = self.hparams
        if not hp.vgae_ckpt_path:
            raise ValueError(
                "CurriculumDataModule requires vgae_ckpt_path — train VGAE autoencoder first"
            )

        # Load datasets via parent — populates _train_ds, _val_ds, properties
        super().setup(stage)

        normals = [g for g in self._train_ds if int(g.y[0]) == 0]
        attacks = [g for g in self._train_ds if int(g.y[0]) == 1]

        device = torch.device("cpu")
        vgae, _ = load_inner_model("vgae", Path(hp.vgae_ckpt_path), device)
        scores = vgae.score_difficulty(normals, canid_weight=hp.canid_weight)
        del vgae
        gc.collect()

        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))

        # Precompute per-graph node counts once — full_dataset is a list of
        # already-materialized Data objects, so .num_nodes is just x.shape[0].
        # CurriculumSampler rebuilds its inner sampler each epoch by indexing
        # this tensor with the active subset (O(M) tensor op, not a walk).
        dataset_sizes = torch.tensor(
            [g.num_nodes for g in full_dataset], dtype=torch.long,
        )

        # Defer VRAM budget to train_dataloader() — model isn't on GPU yet
        # during setup(). CurriculumSampler accepts max_num_nodes=None.
        self._batch_sampler = CurriculumSampler(
            full_dataset, normal_indices, attack_indices, scores,
            batch_size=hp.batch_size, max_epochs=hp.max_epochs,
            curriculum_start_ratio=hp.curriculum_start_ratio,
            curriculum_end_ratio=hp.curriculum_end_ratio,
            difficulty_percentile=hp.difficulty_percentile,
            dataset_sizes=dataset_sizes,
            max_num_nodes=None,
            mean_nodes=1.0,
        )
        self._train_loader = make_graph_loader(
            full_dataset, batch_sampler=self._batch_sampler, num_workers=hp.num_workers,
            device=self._prefetch_device(),
        )
        self._val_loader = None  # built lazily in val_dataloader()

    def _prefetch_device(self):
        """Return GPU device for PrefetchLoader, or None for CPU."""
        trainer = getattr(self, "trainer", None)
        if trainer and torch.cuda.is_available():
            return trainer.strategy.root_device
        return None

    def _build_val_loader(self):
        hp = self.hparams
        bs = max(8, hp.batch_size)
        val_data = list(self._val_ds)
        device = self._prefetch_device()
        if hp.dynamic_batching:
            trainer = getattr(self, "trainer", None)
            model = trainer.lightning_module if trainer else None
            model_hp = getattr(model, "hparams", {}) if model else {}
            result = node_budget(
                hp.dataset, hp.lake_root,
                conv_type=model_hp.get("conv_type", hp.conv_type),
                heads=model_hp.get("heads", hp.heads),
                model=model, train_dataset=val_data,
            )
            nw = hp.num_workers if hp.num_workers is not None else 2
            num_steps = max(1, math.ceil(len(val_data) * result.mean_nodes / result.budget))
            val_sizes = torch.tensor(
                [g.num_nodes for g in val_data], dtype=torch.long,
            )
            sampler = NodeBudgetBatchSampler(
                val_sizes, max_num=result.budget, shuffle=False,
                skip_too_big=True, num_steps=num_steps,
            )
            return make_graph_loader(val_data, batch_sampler=sampler, num_workers=nw, device=device)
        return make_graph_loader(val_data, batch_size=bs, shuffle=False, num_workers=hp.num_workers, device=device)

    def on_train_epoch_start(self, trainer, pl_module):
        if self._batch_sampler is not None:
            self._batch_sampler.set_epoch(trainer.current_epoch)

    def train_dataloader(self):
        hp = self.hparams
        if hp.dynamic_batching and self._batch_sampler.max_num_nodes is None:
            trainer = getattr(self, "trainer", None)
            model = trainer.lightning_module if trainer else None
            model_hp = getattr(model, "hparams", {}) if model else {}
            result = node_budget(
                hp.dataset, hp.lake_root,
                conv_type=model_hp.get("conv_type", hp.conv_type),
                heads=model_hp.get("heads", hp.heads),
                model=model, train_dataset=self._batch_sampler.dataset,
            )
            self._batch_sampler.max_num_nodes = result.budget
            self._batch_sampler.mean_nodes = result.mean_nodes
            self._batch_sampler._inner = self._batch_sampler._build_inner()
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = self._build_val_loader()
        return self._val_loader


# ---------------------------------------------------------------------------
# FusionDataModule — frozen VGAE+GAT state caching
# ---------------------------------------------------------------------------


class FusionDataModule(pl.LightningDataModule):
    """Loads frozen VGAE+GAT, caches state vectors, serves DataLoaders.

    Wraps CAN data loading internally — callers never touch raw graph data.
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

        from graphids.core.models._training import load_inner_model

        cfg_ns = types.SimpleNamespace(
            dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
            preprocessing=types.SimpleNamespace(
                window_size=hp.window_size, stride=hp.stride,
                train_val_split=1.0 - hp.val_fraction,
            ),
        )
        train_ds, val_ds, _ = load_datasets(cfg_ns, CANBusDataset)
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
        from graphids.commands.extract_fusion_states import (
            FUSION_STATES_DIR,
            TRAIN_FILENAME,
            VAL_FILENAME,
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
