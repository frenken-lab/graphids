"""Curriculum learning: VGAE difficulty scoring + tier bucketing + epoch callback."""

from __future__ import annotations

import gc
import math
from pathlib import Path

import torch

from graphids.core.callbacks import CallbackBase


def build_curriculum_tiers(
    train_ds, *, vgae_ckpt_path: str, canid_weight: float,
    num_tiers: int = 10, seed: int = 0,
) -> tuple[torch.Tensor, list[list[int]], list[int], list, torch.Tensor]:
    """Score graphs by VGAE difficulty and bucket normals into tiers.

    Returns (scores, normal_tier_indices, attack_indices, full_dataset, dataset_sizes).
    """
    from graphids.core.models.base import load_inner_model

    if not vgae_ckpt_path:
        raise ValueError("curriculum requires vgae_ckpt_path")

    vgae, _ = load_inner_model("vgae", Path(vgae_ckpt_path), torch.device("cpu"))
    normals = [g for g in train_ds if int(g.y[0]) == 0]
    attacks = [g for g in train_ds if int(g.y[0]) == 1]
    scores = vgae.score_difficulty(normals, canid_weight=canid_weight)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)
    del vgae
    gc.collect()

    full_dataset = normals + attacks
    dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)

    sorted_order = torch.argsort(scores).tolist()
    bucket_size = max(1, math.ceil(len(sorted_order) / num_tiers))
    normal_tier_indices = [sorted_order[i:i + bucket_size] for i in range(0, len(sorted_order), bucket_size)]
    attack_indices = list(range(len(normals), len(full_dataset)))

    return scores, normal_tier_indices, attack_indices, full_dataset, dataset_sizes


class CurriculumEpochCallback(CallbackBase):
    """Select active curriculum tiers each epoch. No-op when datamodule doesn't use tiers."""

    def on_train_epoch_start(self, trainer, model):
        dm = trainer.datamodule
        if getattr(dm, "_tier_batches", None) is not None:
            dm._select_active_tiers(trainer.current_epoch)
