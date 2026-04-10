"""Dataset-agnostic graph DataModule.

``dataset_cls`` is a dotted class-path string resolved via ``importlib``,
so adding a new graph domain is config-only — no subclass needed.

Curriculum learning is a ``sampler="curriculum"`` toggle. When enabled,
``setup()`` scores graphs via VGAE and buckets normals into difficulty
tiers. A ``CurriculumEpochCallback`` selects active tiers each epoch.
"""

from __future__ import annotations

import importlib
import math

import torch
from torch_geometric.data import Batch, InMemoryDataset

from graphids.config.topology import cache_dir, data_dir, load_catalog
from graphids.core.data.budget import autosize_workers, node_budget
from graphids.core.data.sampler import NodeBudgetBatchSampler


def _prefetch(loader, device: torch.device | None):
    if device is not None:
        from torch_geometric.loader import PrefetchLoader
        return PrefetchLoader(loader, device=device)
    return loader


def _spawn_loader(
    dataset, *, batch_size=1, batch_sampler=None, shuffle=False,
    num_workers: int = 0, prefetch_factor: int = 2,
    device: torch.device | None = None,
):
    """PyGDataLoader with spawn/persistent_workers defaults + PrefetchLoader."""
    import torch.multiprocessing as mp
    from torch_geometric.loader import DataLoader as PyGDataLoader

    kw: dict = dict(num_workers=num_workers, pin_memory=device is None)
    if num_workers > 0:
        kw.update(persistent_workers=True, multiprocessing_context="spawn",
                  worker_init_fn=lambda _: mp.set_sharing_strategy("file_system"),
                  prefetch_factor=prefetch_factor)
    if batch_sampler is not None:
        loader = PyGDataLoader(dataset, batch_sampler=batch_sampler, **kw)
    else:
        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kw)
    return _prefetch(loader, device)


def _prebatched_loader(batches, *, shuffle: bool = True, device: torch.device | None = None):
    """TorchDataLoader for pre-collated Batch lists + PrefetchLoader."""
    from torch.utils.data import DataLoader as TorchDataLoader

    loader = TorchDataLoader(
        batches, batch_size=None, shuffle=shuffle,
        collate_fn=lambda x: x.clone() if hasattr(x, "clone") else x,
    )
    return _prefetch(loader, device)


class GraphDataModule:
    """Dataset-agnostic graph DataModule.

    ``dataset_cls`` is a dotted class-path string resolved via importlib.

    Curriculum toggle:
        Pass ``sampler="curriculum"`` + ``vgae_ckpt_path`` to enable
        VGAE-scored difficulty gating. Requires a
        ``CurriculumEpochCallback`` in the trainer's callback list.
    """

    def __init__(
        self,
        dataset: str,
        dataset_cls: str = "graphids.core.data.datasets.can_bus.CANBusDataset",
        lake_root: str | None = None,
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
        if lake_root is None:
            from graphids.config.settings import get_settings
            lake_root = get_settings().lake_root
        # Store init args as a dict for downstream kwargs access.
        self.hparams = {k: v for k, v in locals().items() if k != "self"}
        module_path, cls_name = dataset_cls.rsplit(".", 1)
        self._dataset_cls: type[InMemoryDataset] = getattr(
            importlib.import_module(module_path), cls_name
        )
        self._train_ds: InMemoryDataset | None = None
        self._val_ds: InMemoryDataset | None = None
        self._test_datasets: dict[str, InMemoryDataset] = {}
        self._prebatched_train: list[Batch] | None = None
        # Curriculum: populated by _setup_curriculum, pre-batched on first epoch
        self._tier_graphs: list[list] | None = None  # per-tier graph lists
        self._tier_sizes: list[torch.Tensor] | None = None  # per-tier node counts
        self._tier_batches: list[list[Batch]] | None = None  # pre-batched tiers
        self._active_batches: list[Batch] | None = None
        # Device for PrefetchLoader — set by trainer via _set_device()
        self._device: torch.device | None = None

    def _set_device(self, device: torch.device | None) -> None:
        """Called by Trainer to tell the datamodule which device to prefetch to."""
        self._device = device

    def setup(self, stage: str | None = None) -> None:
        if self._train_ds is not None:
            return
        hp = self.hparams
        self._train_ds, self._val_ds, self._test_datasets = self._load_datasets()
        if hp["sampler"] == "curriculum":
            self._setup_curriculum()

    def _load_datasets(self):
        """Load train/val/test datasets from cache."""
        hp = self.hparams
        root = cache_dir(hp["lake_root"], hp["dataset"])
        raw = data_dir(hp["lake_root"], hp["dataset"])
        common = dict(
            window_size=hp["window_size"], stride=hp["stride"],
            val_fraction=hp["val_fraction"], seed=hp["seed"],
        )
        train_ds = self._dataset_cls(root=root, raw_dir=raw, split="train", **common)
        val_ds = self._dataset_cls(root=root, raw_dir=raw, split="val", **common)

        test_datasets = {}
        for subdir in load_catalog()[hp["dataset"]].get("test_subdirs", []):
            test_raw = raw / subdir
            if test_raw.exists():
                test_datasets[subdir] = self._dataset_cls(
                    root=root, raw_dir=test_raw, split="test", **common,
                )
        return train_ds, val_ds, test_datasets

    def _setup_curriculum(self) -> None:
        """Score graphs by VGAE difficulty, bucket into tiers + attack tier."""
        from graphids.core.data.curriculum import build_curriculum_tiers

        hp = self.hparams
        scores, normal_tiers, attack_indices, full_dataset, dataset_sizes = (
            build_curriculum_tiers(
                self._train_ds,
                vgae_ckpt_path=hp["vgae_ckpt_path"],
                canid_weight=hp["canid_weight"],
                num_tiers=hp.get("num_tiers", 10),
                seed=hp["seed"],
            )
        )
        # Build per-tier graph lists + size tensors for _prebatch
        self._tier_graphs = []
        self._tier_sizes = []
        for tier_idx in normal_tiers:
            self._tier_graphs.append([full_dataset[i] for i in tier_idx])
            self._tier_sizes.append(dataset_sizes[tier_idx])
        # Attack tier (always active)
        if attack_indices:
            self._tier_graphs.append([full_dataset[i] for i in attack_indices])
            self._tier_sizes.append(dataset_sizes[attack_indices])

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
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return max(2, int(ds._data.y.unique().numel()))

    @property
    def edge_dim(self) -> int:
        ds = self._train_ds or next(iter(self._test_datasets.values()), None)
        assert ds is not None, "call setup() first"
        return ds[0].edge_attr.shape[1]

    # -- Shared helpers -------------------------------------------------------

    def _budget_result(self, dataset=None):
        """Probe VRAM budget for the given dataset (or train_ds)."""
        hp = self.hparams
        return node_budget(
            hp["dataset"], hp["lake_root"],
            conv_type=hp.get("conv_type", "gatv2"),
            heads=hp.get("heads", 4),
            train_dataset=dataset or self._train_ds,
        )

    def _prebatch(self, graphs, sizes) -> list[Batch]:
        """Pre-collate graphs into Batches using NodeBudgetBatchSampler."""
        result = self._budget_result(graphs)
        num_steps = max(1, math.ceil(len(graphs) * result.mean_nodes / result.budget))
        sampler = NodeBudgetBatchSampler(
            sizes, max_num=result.budget,
            shuffle=False, skip_too_big=True, num_steps=num_steps,
        )
        return [Batch.from_data_list([graphs[i] for i in plan]) for plan in sampler]

    def _prefetch_device(self):
        return self._device if torch.cuda.is_available() else None

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

    def _build_loader(self, dataset, shuffle: bool):
        hp = self.hparams
        nw = hp.get("num_workers")
        pf = hp.get("prefetch_factor", 2)
        device = self._prefetch_device()

        if not hp["dynamic_batching"]:
            return _spawn_loader(
                dataset, batch_size=max(8, hp["batch_size"]),
                shuffle=shuffle, num_workers=nw if nw is not None else 2,
                device=device, prefetch_factor=pf,
            )

        result = self._budget_result(dataset)
        if nw is None:
            nw, pf = autosize_workers(None, dataset, result, default_prefetch=pf)

        num_steps = max(1, math.ceil(len(dataset) * result.mean_nodes / result.budget))
        sampler = NodeBudgetBatchSampler(
            dataset.num_nodes_per_graph, max_num=result.budget,
            shuffle=shuffle, skip_too_big=True, num_steps=num_steps,
        )
        return _spawn_loader(
            dataset, batch_sampler=sampler,
            num_workers=nw, device=device, prefetch_factor=pf,
        )

    def _prebatched_train_dataloader(self):
        """Pre-batched training: collate once, shuffle batch order per epoch."""
        if self._prebatched_train is None:
            self._prebatched_train = self._prebatch(
                self._train_ds, self._train_ds.num_nodes_per_graph,
            )
        return _prebatched_loader(
            self._prebatched_train, shuffle=True, device=self._prefetch_device(),
        )

    def _curriculum_train_dataloader(self):
        """Tier-based curriculum: pre-batch each tier once, select active per epoch."""
        if self._tier_batches is None:
            self._tier_batches = [
                self._prebatch(graphs, sizes)
                for graphs, sizes in zip(self._tier_graphs, self._tier_sizes)
            ]
            self._select_active_tiers(0)
        return _prebatched_loader(
            self._active_batches, shuffle=True, device=self._prefetch_device(),
        )

    def _select_active_tiers(self, epoch: int) -> None:
        """Select difficulty tiers based on curriculum ratio.

        Called by ``CurriculumEpochCallback.on_train_epoch_start``.
        Tier 0 = easiest, last tier = attacks (always active).
        """
        hp = self.hparams
        ratio = hp["curriculum_start_ratio"] + (
            hp["curriculum_end_ratio"] - hp["curriculum_start_ratio"]
        ) * min(epoch / max(hp["max_epochs"] - 1, 1), 1.0)
        n_normal = len(self._tier_batches) - 1  # last tier is attacks
        active_count = max(1, min(n_normal, math.ceil(
            ratio * n_normal / hp["curriculum_end_ratio"],
        )))
        active: list[Batch] = []
        for i in range(active_count):
            active.extend(self._tier_batches[i])
        active.extend(self._tier_batches[-1])  # attacks always active
        self._active_batches = active
