"""Curriculum-ordered sampling over CAN data.

Extends ``CANBusDataModule`` to add VGAE-driven difficulty scoring and
curriculum scheduling via ``CurriculumSampler``. The VGAE checkpoint is
loaded in ``setup()`` (pre-device-placement); the VRAM-aware node budget
is deferred to ``train_dataloader()`` because the Lightning model is not
on GPU yet when ``setup()`` runs.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import torch

from graphids.core.preprocessing.budget import node_budget
from graphids.core.preprocessing.sampler import CurriculumSampler, make_graph_loader

from .can_bus import CANBusDataModule


class CurriculumDataModule(CANBusDataModule):
    """Curriculum learning with persistent workers.

    Subclasses ``CANBusDataModule`` for data loading + properties
    (``num_ids``, ``in_channels``, ``num_classes``). Adds VGAE difficulty
    scoring and curriculum-ordered batching via ``CurriculumSampler``.
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
            self._batch_sampler.set_node_budget(result.budget, result.mean_nodes)
        return self._train_loader

    def val_dataloader(self):
        if self._val_loader is None:
            self._val_loader = super()._build_loader(self._val_ds, shuffle=False)
        return self._val_loader
