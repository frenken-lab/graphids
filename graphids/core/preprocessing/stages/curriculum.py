"""Curriculum learning: difficulty-ordered batching for GAT training."""

from __future__ import annotations

import math
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DynamicBatchSampler

from graphids.core.preprocessing.budget import node_budget
from graphids.core.preprocessing.datamodule import (
    CANBusDataModule,
    make_graph_loader,
)


class CurriculumSampler:
    """Curriculum selection — picks which graphs are active each epoch.

    set_epoch() updates active indices based on difficulty scores and
    epoch progress. Wraps a DynamicBatchSampler (or fixed-size batching)
    rebuilt each epoch from the active subset.
    """

    def __init__(
        self,
        dataset,
        normal_indices,
        attack_indices,
        scores,
        *,
        batch_size: int,
        max_epochs: int,
        curriculum_start_ratio: float,
        curriculum_end_ratio: float,
        difficulty_percentile: float,
        max_num_nodes: int | None = None,
        mean_nodes: float = 1.0,
    ):
        assert len(scores) == len(normal_indices)
        self.dataset = dataset
        self.normal_indices = torch.tensor(normal_indices, dtype=torch.long)
        self.attack_indices = attack_indices
        self.scores = torch.tensor(scores) if scores else None
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.curriculum_start_ratio = curriculum_start_ratio
        self.curriculum_end_ratio = curriculum_end_ratio
        self.difficulty_percentile = difficulty_percentile
        self.max_num_nodes = max_num_nodes
        self.mean_nodes = mean_nodes
        self._active_indices = normal_indices + attack_indices
        self._inner = self._build_inner()

    def _build_inner(self):
        if self.max_num_nodes is None:
            return None
        num_steps = max(1, math.ceil(len(self._active_indices) * self.mean_nodes / self.max_num_nodes))
        return DynamicBatchSampler(
            Subset(self.dataset, self._active_indices),
            max_num=self.max_num_nodes, mode="node", shuffle=True,
            skip_too_big=True, num_steps=num_steps,
        )

    def set_epoch(self, epoch: int) -> None:
        if self.scores is None or len(self.scores) == 0:
            return
        ratio = self.curriculum_start_ratio + (
            self.curriculum_end_ratio - self.curriculum_start_ratio
        ) * min(epoch / max(self.max_epochs - 1, 1), 1.0)
        n_normal = min(max(1, int(len(self.normal_indices) * ratio)), len(self.normal_indices))
        threshold = torch.quantile(self.scores, self.difficulty_percentile / 100.0)
        easy = self.scores <= threshold
        hard = ~easy
        easy_idx = self.normal_indices[easy][:n_normal].tolist()
        hard_idx = self.normal_indices[hard].tolist()
        self._active_indices = easy_idx + hard_idx + self.attack_indices
        self._inner = self._build_inner()

    def __iter__(self):
        if self._inner is not None:
            yield from self._inner
        else:
            perm = torch.randperm(len(self._active_indices))
            bs = max(8, self.batch_size)
            for start in range(0, len(perm), bs):
                yield [self._active_indices[perm[j]] for j in range(start, min(start + bs, len(perm)))]

    def __len__(self) -> int:
        if self._inner is not None:
            return len(self._inner)
        bs = max(8, self.batch_size)
        return max(1, (len(self._active_indices) + bs - 1) // bs)


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
        num_workers: int = 2,
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
            num_workers=num_workers, window_size=window_size, stride=stride,
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
        import gc
        from pathlib import Path
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

        # Defer VRAM budget to train_dataloader() — model isn't on GPU yet
        # during setup(). CurriculumSampler accepts max_num_nodes=None.
        self._batch_sampler = CurriculumSampler(
            full_dataset, normal_indices, attack_indices, scores,
            batch_size=hp.batch_size, max_epochs=hp.max_epochs,
            curriculum_start_ratio=hp.curriculum_start_ratio,
            curriculum_end_ratio=hp.curriculum_end_ratio,
            difficulty_percentile=hp.difficulty_percentile,
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
        import torch
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
                num_workers=hp.num_workers,
            )
            num_steps = max(1, math.ceil(len(val_data) * result.mean_nodes / result.budget))
            sampler = DynamicBatchSampler(
                val_data, max_num=result.budget, mode="node", shuffle=False,
                skip_too_big=True, num_steps=num_steps,
            )
            return make_graph_loader(val_data, batch_sampler=sampler, num_workers=hp.num_workers, device=device)
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
                num_workers=hp.num_workers,
            )
            self._batch_sampler.max_num_nodes = result.budget
            self._batch_sampler.mean_nodes = result.mean_nodes
            self._batch_sampler._inner = self._batch_sampler._build_inner()
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = self._build_val_loader()
        return self._val_loader
