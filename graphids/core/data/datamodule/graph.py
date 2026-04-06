"""Dataset-agnostic graph LightningDataModule base + shared dataset loader.

``GraphDataModule`` is the extension point for graph-shaped datasets. Each
concrete dataset (CAN bus, future Ethernet, etc.) lives in its own sibling
module and inherits from this class, setting ``dataset_cls`` to bind the
underlying ``InMemoryDataset`` subclass.

``load_datasets`` is a free helper used by both the graph family and
``FusionDataModule`` — it operates on any ``InMemoryDataset`` that accepts
the ``(root, raw_dir, split, window_size, stride, val_fraction, seed)``
constructor signature and has a catalog entry.

Curriculum learning is a toggle on this module (``sampler="curriculum"``),
not a separate DataModule subclass. When enabled, ``setup()`` loads a VGAE
checkpoint and scores training graphs by difficulty, then
``train_dataloader()`` uses a ``CurriculumSampler`` instead of a plain
``NodeBudgetBatchSampler``. A ``CurriculumEpochCallback`` (in
``graphids.core.data.sampler``) advances the sampler's epoch counter.
"""

from __future__ import annotations

import math
import os
from typing import ClassVar

import pytorch_lightning as pl
import torch
from torch_geometric.data import InMemoryDataset

from graphids.config.paths import cache_dir, data_dir, load_catalog
from graphids.core.data.budget import autosize_workers, node_budget
from graphids.core.data.sampler import CurriculumSampler, NodeBudgetBatchSampler, make_graph_loader


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

    Subclasses set ``dataset_cls`` to a concrete ``InMemoryDataset`` subclass.
    All batching, VRAM sizing, and loader construction logic is shared.

    Curriculum toggle:
        Pass ``sampler="curriculum"`` + ``vgae_ckpt_path`` to enable
        VGAE-scored difficulty gating. The curriculum ratio ramps from
        ``curriculum_start_ratio`` → ``curriculum_end_ratio`` over
        ``max_epochs``. Requires a ``CurriculumEpochCallback`` in the
        trainer's callback list to advance the epoch counter.
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
        # --- curriculum toggle ---
        sampler: str = "standard",
        vgae_ckpt_path: str = "",
        curriculum_start_ratio: float = 1.0,
        curriculum_end_ratio: float = 10.0,
        difficulty_percentile: float = 75.0,
        max_epochs: int = 300,
        canid_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}
        # Curriculum state — populated by _setup_curriculum() when sampler="curriculum"
        self._curriculum_sampler: CurriculumSampler | None = None
        self._curriculum_dataset: list | None = None
        self._curriculum_loader = None

    def setup(self, stage: str | None = None) -> None:
        hp = self.hparams
        self._train_ds, self._val_ds, self._test_datasets = load_datasets(
            dataset=hp["dataset"],
            lake_root=hp["lake_root"],
            seed=hp["seed"],
            window_size=hp["window_size"],
            stride=hp["stride"],
            train_val_split=1.0 - hp["val_fraction"],
            dataset_cls=self.dataset_cls,
        )
        if hp["sampler"] == "curriculum":
            self._setup_curriculum()

    def _setup_curriculum(self) -> None:
        """Score per-graph difficulty with VGAE and build the curriculum sampler.

        Runs once during ``setup()``. Loads the VGAE checkpoint on CPU,
        scores normal-class training graphs, then discards the model.
        The resulting sampler starts with ``max_num_nodes=None`` — the real
        VRAM budget is deferred to ``_curriculum_train_dataloader()`` because
        the Lightning model isn't on GPU yet when ``setup()`` runs.
        """
        import gc
        from pathlib import Path

        from graphids.core.models._training import load_inner_model

        hp = self.hparams
        if not hp["vgae_ckpt_path"]:
            raise ValueError(
                "sampler='curriculum' requires vgae_ckpt_path — train a VGAE autoencoder first"
            )

        device = torch.device("cpu")
        vgae, _ = load_inner_model("vgae", Path(hp["vgae_ckpt_path"]), device)

        normals = [g for g in self._train_ds if int(g.y[0]) == 0]
        attacks = [g for g in self._train_ds if int(g.y[0]) == 1]
        scores = vgae.score_difficulty(normals, canid_weight=hp["canid_weight"])
        del vgae
        gc.collect()

        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))
        dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)

        self._curriculum_dataset = full_dataset
        self._curriculum_sampler = CurriculumSampler(
            full_dataset,
            normal_indices,
            attack_indices,
            scores,
            batch_size=hp["batch_size"],
            max_epochs=hp["max_epochs"],
            curriculum_start_ratio=hp["curriculum_start_ratio"],
            curriculum_end_ratio=hp["curriculum_end_ratio"],
            difficulty_percentile=hp["difficulty_percentile"],
            dataset_sizes=dataset_sizes,
            max_num_nodes=None,
            mean_nodes=1.0,
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

    def _curriculum_train_dataloader(self):
        """Build the curriculum training loader (first call only).

        Deferred to the first ``train_dataloader()`` call because
        ``setup()`` runs before the model is on GPU, so ``node_budget``
        can't probe VRAM during setup. By the time Lightning calls
        ``train_dataloader()``, the model is on-device and the probe works.
        """
        if self._curriculum_loader is not None:
            return self._curriculum_loader

        hp = self.hparams

        # Finalize budget now that model is on GPU
        if hp["dynamic_batching"] and self._curriculum_sampler.max_num_nodes is None:
            model, model_hp = self._model_and_hp()
            result = node_budget(
                hp["dataset"],
                hp["lake_root"],
                conv_type=model_hp.get("conv_type", hp.get("conv_type", "gatv2")),
                heads=model_hp.get("heads", hp.get("heads", 4)),
                model=model,
                train_dataset=self._curriculum_dataset,
            )
            self._curriculum_sampler.set_node_budget(result.budget, result.mean_nodes)

        nw = hp.get("num_workers")
        pf = hp.get("prefetch_factor", 2)
        self._curriculum_loader = make_graph_loader(
            self._curriculum_dataset,
            batch_sampler=self._curriculum_sampler,
            num_workers=nw if nw is not None else 2,
            device=self._prefetch_device(),
            prefetch_factor=pf if (nw or 2) > 0 else None,
        )
        return self._curriculum_loader
