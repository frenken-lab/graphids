"""Dataset-agnostic graph LightningDataModule + shared dataset loader.

``GraphDataModule`` accepts a ``dataset_cls`` class-path string that is
resolved at init time via ``importlib``, so the dataset domain is fully
config-driven — no subclass needed per domain.

``load_datasets`` is a free helper used by both the graph family and
``FusionDataModule`` — it operates on any ``InMemoryDataset`` that accepts
the ``(root, raw_dir, split, window_size, stride, val_fraction, seed)``
constructor signature and has a catalog entry.

Curriculum learning is a toggle on this module (``sampler="curriculum"``),
not a separate DataModule subclass. When enabled, ``setup()`` loads a VGAE
checkpoint, scores training graphs by difficulty, and buckets normals into
K difficulty tiers (tier 0 = easiest). Each tier is pre-batched once.
A ``CurriculumEpochCallback`` selects which tiers are active each epoch
— O(1) epoch transition, zero re-collation.
"""

from __future__ import annotations

import importlib
import math
import os

import pytorch_lightning as pl
import torch
from torch_geometric.data import InMemoryDataset

from graphids.config.paths import cache_dir, data_dir, load_catalog
from graphids.core.data.budget import autosize_workers, node_budget
from graphids.core.data.sampler import (
    NodeBudgetBatchSampler,
    make_graph_loader,
)


def load_datasets(
    *,
    dataset: str,
    lake_root: str,
    seed: int,
    window_size: int,
    stride: int,
    train_val_split: float,
    dataset_cls: type[InMemoryDataset],
) -> tuple[InMemoryDataset, InMemoryDataset, dict[str, InMemoryDataset]]:
    """Load train/val/test datasets from cache using the given dataset class.

    Returns (train_ds, val_ds, {name: test_ds}).
    """
    root = cache_dir(lake_root, dataset)
    raw = data_dir(lake_root, dataset)
    common = dict(
        window_size=window_size,
        stride=stride,
        val_fraction=1.0 - train_val_split,
        seed=seed,
    )
    train_ds = dataset_cls(root=root, raw_dir=raw, split="train", **common)
    val_ds = dataset_cls(root=root, raw_dir=raw, split="val", **common)

    test_datasets = {}
    catalog = load_catalog()
    entry = catalog[dataset]
    for subdir in entry.get("test_subdirs", []):
        test_raw = raw / subdir
        if test_raw.exists():
            test_datasets[subdir] = dataset_cls(root=root, raw_dir=test_raw, split="test", **common)

    return train_ds, val_ds, test_datasets


class GraphDataModule(pl.LightningDataModule):
    """Dataset-agnostic graph DataModule.

    ``dataset_cls`` is a dotted class-path string (e.g.
    ``"graphids.core.data.datasets.can_bus.CANBusDataset"``) resolved via
    importlib at init time. This keeps the dataset domain fully config-driven.

    Curriculum toggle:
        Pass ``sampler="curriculum"`` + ``vgae_ckpt_path`` to enable
        VGAE-scored difficulty gating. The curriculum ratio ramps from
        ``curriculum_start_ratio`` → ``curriculum_end_ratio`` over
        ``max_epochs``. Requires a ``CurriculumEpochCallback`` in the
        trainer's callback list to advance the epoch counter.
    """

    def __init__(
        self,
        dataset: str,
        dataset_cls: str = "graphids.core.data.datasets.can_bus.CANBusDataset",
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
        # --- curriculum toggle ---
        sampler: str = "standard",
        vgae_ckpt_path: str = "",
        curriculum_start_ratio: float = 1.0,
        curriculum_end_ratio: float = 10.0,
        max_epochs: int = 300,
        canid_weight: float = 0.1,
        num_tiers: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Resolve dataset_cls class-path string → actual class
        module_path, cls_name = dataset_cls.rsplit(".", 1)
        self._dataset_cls: type[InMemoryDataset] = getattr(
            importlib.import_module(module_path), cls_name
        )
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}
        # Pre-batched train set — built once in first train_dataloader() call
        self._prebatched_train = None
        # Curriculum tier state — populated by _setup_curriculum()
        self._curriculum_full_dataset = None
        self._curriculum_dataset_sizes = None
        self._curriculum_normal_tiers = None  # list[list[int]], sorted by difficulty
        self._curriculum_attack_indices = None
        self._tier_batches = None  # list[list[Batch]], one per normal tier
        self._attack_tier_batches = None  # list[Batch]
        self._active_batches = None  # concatenated active tiers for current epoch

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return  # datasets pre-injected (e.g. by Monarch actor)
        hp = self.hparams
        self._train_ds, self._val_ds, self._test_datasets = load_datasets(
            dataset=hp["dataset"],
            lake_root=hp["lake_root"],
            seed=hp["seed"],
            window_size=hp["window_size"],
            stride=hp["stride"],
            train_val_split=1.0 - hp["val_fraction"],
            dataset_cls=self._dataset_cls,
        )
        if hp["sampler"] == "curriculum":
            self._setup_curriculum()

    def _setup_curriculum(self) -> None:
        """Score graphs and bucket into difficulty tiers.

        Pre-batching is deferred to the first ``train_dataloader()`` call
        because ``node_budget()`` needs the model on GPU.
        """
        from graphids.core.data.sampler import build_curriculum_tiers

        hp = self.hparams
        scores, normal_tiers, attack_indices, full_dataset, dataset_sizes = build_curriculum_tiers(
            self._train_ds,
            vgae_ckpt_path=hp["vgae_ckpt_path"],
            canid_weight=hp["canid_weight"],
            num_tiers=hp.get("num_tiers", 10),
            seed=hp["seed"],
        )
        self._curriculum_full_dataset = full_dataset
        self._curriculum_dataset_sizes = dataset_sizes
        self._curriculum_normal_tiers = normal_tiers
        self._curriculum_attack_indices = attack_indices

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
        return ds[0].x.shape[1]

    @property
    def num_classes(self) -> int:
        """Number of unique target classes across the dataset (floor at 2 for binary)."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return max(2, int(ds._data.y.unique().numel()))

    @property
    def edge_dim(self) -> int:
        """Edge feature dimensionality."""
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1]

    # -- DataLoaders ----------------------------------------------------------

    def train_dataloader(self):
        if self.hparams["sampler"] == "curriculum":
            return self._curriculum_train_dataloader()
        if self.hparams["dynamic_batching"]:
            return self._prebatched_train_dataloader()
        return self._build_loader(self._train_ds, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self._val_ds, shuffle=False)

    def test_dataloader(self):
        return [self._build_loader(ds, shuffle=False) for ds in self._test_datasets.values()]

    def _prefetch_device(self):
        """Return GPU device for PrefetchLoader async H2D, or None for CPU."""
        trainer = getattr(self, "trainer", None)
        if trainer and torch.cuda.is_available():
            return trainer.strategy.root_device
        return None

    def _model_and_hp(self):
        """Read model + hparams from trainer (available after device placement)."""
        trainer = getattr(self, "trainer", None)
        model = trainer.lightning_module if trainer else None
        model_hp = getattr(model, "hparams", {}) if model else {}
        return model, model_hp

    def _build_loader(self, dataset, shuffle: bool):
        hp = self.hparams
        nw = hp.get("num_workers")
        pf = hp.get("prefetch_factor", 2)
        device = self._prefetch_device()

        if not hp["dynamic_batching"]:
            return make_graph_loader(
                dataset,
                batch_size=max(8, hp["batch_size"]),
                shuffle=shuffle,
                num_workers=nw if nw is not None else 2,
                device=device,
                prefetch_factor=pf if (nw or 2) > 0 else None,
            )

        model, model_hp = self._model_and_hp()
        result = node_budget(
            hp["dataset"],
            hp["lake_root"],
            conv_type=model_hp.get("conv_type", hp.get("conv_type", "gatv2")),
            heads=model_hp.get("heads", hp.get("heads", 4)),
            model=model,
            train_dataset=dataset,
        )
        if nw is None:
            nw, pf = autosize_workers(model, dataset, result, default_prefetch_factor=pf)

        num_steps = max(1, math.ceil(len(dataset) * result.mean_nodes / result.budget))
        sampler = NodeBudgetBatchSampler(
            dataset.num_nodes_per_graph,
            max_num=result.budget,
            shuffle=shuffle,
            skip_too_big=True,
            num_steps=num_steps,
        )
        return make_graph_loader(
            dataset,
            batch_sampler=sampler,
            num_workers=nw,
            device=device,
            prefetch_factor=pf if nw > 0 else None,
        )

    def _prebatched_train_dataloader(self):
        """Pre-batched training loader (standard path, dynamic batching).

        First call: probe VRAM budget, plan batches, pre-collate all Batches.
        Subsequent calls: return loader over cached list of Batches.
        DataLoader shuffles batch *order* per epoch; graph-to-batch assignment
        is fixed (acceptable — packing is deterministic for a given budget).
        """
        if self._prebatched_train is None:
            from torch_geometric.data import Batch

            hp = self.hparams
            model, model_hp = self._model_and_hp()
            result = node_budget(
                hp["dataset"],
                hp["lake_root"],
                conv_type=model_hp.get("conv_type", hp.get("conv_type", "gatv2")),
                heads=model_hp.get("heads", hp.get("heads", 4)),
                model=model,
                train_dataset=self._train_ds,
            )
            num_steps = max(1, math.ceil(len(self._train_ds) * result.mean_nodes / result.budget))
            sampler = NodeBudgetBatchSampler(
                self._train_ds.num_nodes_per_graph,
                max_num=result.budget,
                shuffle=False,  # deterministic packing; shuffle batch ORDER below
                skip_too_big=True,
                num_steps=num_steps,
            )
            plans = list(sampler)
            self._prebatched_train = [
                Batch.from_data_list([self._train_ds[i] for i in plan]) for plan in plans
            ]

        return make_graph_loader(
            self._prebatched_train,
            batch_size=None,
            shuffle=True,
            num_workers=0,  # O(1) __getitem__; workers add IPC overhead for no gain
            device=self._prefetch_device(),
        )

    def _prebatch_tiers(self) -> None:
        """Pre-batch each curriculum tier (called once, deferred to first epoch).

        Deferred because ``node_budget()`` needs the model on GPU.
        """
        from torch_geometric.data import Batch

        hp = self.hparams
        model, model_hp = self._model_and_hp()
        result = node_budget(
            hp["dataset"],
            hp["lake_root"],
            conv_type=model_hp.get("conv_type", hp.get("conv_type", "gatv2")),
            heads=model_hp.get("heads", hp.get("heads", 4)),
            model=model,
            train_dataset=self._curriculum_full_dataset,
        )
        ds = self._curriculum_full_dataset
        sizes = self._curriculum_dataset_sizes

        # Pre-batch each normal tier
        self._tier_batches = []
        for tier_indices in self._curriculum_normal_tiers:
            tier_sizes = sizes[tier_indices]
            num_steps = max(1, math.ceil(len(tier_indices) * result.mean_nodes / result.budget))
            sampler = NodeBudgetBatchSampler(
                tier_sizes,
                max_num=result.budget,
                shuffle=False,
                skip_too_big=True,
                num_steps=num_steps,
            )
            plans = list(sampler)
            self._tier_batches.append(
                [Batch.from_data_list([ds[tier_indices[i]] for i in plan]) for plan in plans]
            )

        # Pre-batch attack tier (always active)
        atk = self._curriculum_attack_indices
        if atk:
            atk_sizes = sizes[atk]
            num_steps = max(1, math.ceil(len(atk) * result.mean_nodes / result.budget))
            sampler = NodeBudgetBatchSampler(
                atk_sizes,
                max_num=result.budget,
                shuffle=False,
                skip_too_big=True,
                num_steps=num_steps,
            )
            plans = list(sampler)
            self._attack_tier_batches = [
                Batch.from_data_list([ds[atk[i]] for i in plan]) for plan in plans
            ]
        else:
            self._attack_tier_batches = []

    def _select_active_tiers(self, epoch: int) -> None:
        """Select which difficulty tiers are active based on curriculum ratio.

        Called by ``CurriculumEpochCallback.on_train_epoch_start`` each epoch.
        Tier 0 = easiest, tier K-1 = hardest.  Early epochs include fewer
        tiers; later epochs include all.  Attacks are always included.
        """
        hp = self.hparams
        ratio = hp["curriculum_start_ratio"] + (
            hp["curriculum_end_ratio"] - hp["curriculum_start_ratio"]
        ) * min(epoch / max(hp["max_epochs"] - 1, 1), 1.0)
        num_tiers = len(self._tier_batches)
        active_count = max(
            1,
            min(num_tiers, math.ceil(ratio * num_tiers / hp["curriculum_end_ratio"])),
        )
        active: list = []
        for i in range(active_count):
            active.extend(self._tier_batches[i])
        active.extend(self._attack_tier_batches)
        self._active_batches = active

    def _curriculum_train_dataloader(self):
        """Tier-based curriculum training loader.

        First call: probe VRAM budget, pre-batch each tier.
        Every call: return loader over active tiers (set by callback).
        """
        if self._tier_batches is None:
            self._prebatch_tiers()
            self._select_active_tiers(0)

        return make_graph_loader(
            self._active_batches,
            batch_size=None,
            shuffle=True,
            num_workers=0,
            device=self._prefetch_device(),
        )
