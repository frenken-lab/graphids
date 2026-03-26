"""Curriculum learning: difficulty-ordered batching for GAT training."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DynamicBatchSampler

from graphids.core.models._training import compute_node_budget
from graphids.core.preprocessing.datamodule import make_graph_loader


class CurriculumSampler:
    """Curriculum selection — picks which graphs are active each epoch.

    set_epoch() updates active indices based on difficulty scores and
    epoch progress. Wraps a DynamicBatchSampler (or fixed-size batching)
    rebuilt each epoch from the active subset.
    """

    def __init__(self, dataset, normal_indices, attack_indices, scores, cfg, max_num_nodes=None):
        assert len(scores) == len(normal_indices)
        self.dataset = dataset
        self.normal_indices = torch.tensor(normal_indices, dtype=torch.long)
        self.attack_indices = attack_indices
        self.scores = torch.tensor(scores) if scores else None
        self.cfg = cfg
        self.max_num_nodes = max_num_nodes
        self._active_indices = normal_indices + attack_indices
        self._inner = self._build_inner()

    def _build_inner(self):
        if self.max_num_nodes is None:
            return None
        subset = Subset(self.dataset, self._active_indices)
        sampler = DynamicBatchSampler(subset, max_num=self.max_num_nodes, mode="node", shuffle=True, skip_too_big=True)
        return sampler

    def set_epoch(self, epoch: int) -> None:
        """Update active indices for curriculum progression."""
        t = self.cfg.training
        progress = min(epoch / max(t.max_epochs, 1), 1.0)
        ratio = t.curriculum_start_ratio + (t.curriculum_end_ratio - t.curriculum_start_ratio) * progress
        percentile = t.difficulty_percentile + (95.0 - t.difficulty_percentile) * progress

        # Filter normals by difficulty threshold, subsample to ratio
        if self.scores is not None:
            threshold = self.scores.quantile(percentile / 100).item()
            mask = self.scores >= threshold
            hard = self.normal_indices[mask].tolist() or self.normal_indices.tolist()
        else:
            hard = self.normal_indices.tolist()

        n = min(int(len(self.attack_indices) * ratio), len(hard))
        selected = [hard[i] for i in torch.randperm(len(hard))[:n].tolist()] if 0 < n < len(hard) else hard
        self._active_indices = selected + self.attack_indices
        self._inner = self._build_inner()

    def __iter__(self):
        if self._inner is not None:
            yield from self._inner
        else:
            bs = max(8, self.cfg.training.batch_size)
            perm = torch.randperm(len(self._active_indices)).tolist()
            for start in range(0, len(perm), bs):
                yield [self._active_indices[perm[j]] for j in range(start, min(start + bs, len(perm)))]

    def __len__(self) -> int:
        if self._inner is not None:
            return len(self._inner)
        bs = max(8, self.cfg.training.batch_size)
        return max(1, (len(self._active_indices) + bs - 1) // bs)


class CurriculumDataModule(pl.LightningDataModule):
    """Curriculum learning with persistent workers.

    Rebuilds batch sampler each epoch via set_epoch() to control
    which graphs are yielded based on difficulty progression.
    """

    def __init__(self, normals, attacks, scores, val_data, cfg):
        super().__init__()
        self.val_data = val_data
        self.cfg = cfg
        self._current_epoch = 0

        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))

        nw = cfg.num_workers
        budget = None
        if cfg.training.dynamic_batching:
            bs = max(8, cfg.training.batch_size)
            info = compute_node_budget(bs, cfg, conv_type=cfg.gat.conv_type, heads=cfg.gat.heads)
            budget = info.budget

        self._batch_sampler = CurriculumSampler(
            full_dataset, normal_indices, attack_indices, scores, cfg, budget,
        )
        self._train_loader = make_graph_loader(
            full_dataset, batch_sampler=self._batch_sampler, num_workers=nw,
        )

    def train_dataloader(self):
        self._batch_sampler.set_epoch(self._current_epoch)
        self._current_epoch += 1
        return self._train_loader

    def val_dataloader(self):
        bs = max(8, self.cfg.training.batch_size)
        nw = self.cfg.num_workers

        if self.cfg.training.dynamic_batching:
            info = compute_node_budget(bs, self.cfg, conv_type=self.cfg.gat.conv_type, heads=self.cfg.gat.heads)
            sampler = DynamicBatchSampler(self.val_data, max_num=info.budget, mode="node", shuffle=False, skip_too_big=True)
            return make_graph_loader(self.val_data, batch_sampler=sampler, num_workers=nw)

        return make_graph_loader(self.val_data, batch_size=bs, shuffle=False, num_workers=nw)
