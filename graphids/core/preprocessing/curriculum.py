"""Curriculum learning: difficulty-ordered batching for GAT training."""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch

from graphids.core.models._utils import compute_node_budget


class CurriculumDynamicBatchSampler:
    """Curriculum selection + node-budget packing in a single batch sampler.

    Runs in the main process. Workers receive index batches via queues,
    so set_epoch() mutations propagate even with persistent_workers=True + spawn.
    """

    def __init__(self, dataset, normal_indices, attack_indices, scores, cfg, max_num_nodes=None):
        assert len(scores) == len(normal_indices)
        self.dataset = dataset
        self.normal_indices = normal_indices
        self.attack_indices = attack_indices
        self.scores = scores
        self.cfg = cfg
        self.max_num_nodes = max_num_nodes
        self._active_indices = normal_indices + attack_indices
        self._node_counts = [dataset[i].num_nodes for i in range(len(dataset))]
        self._cached_len: int | None = None

    def set_epoch(self, epoch: int) -> None:
        """Update active indices for curriculum progression."""
        cfg = self.cfg
        progress = min(epoch / max(cfg.training.max_epochs, 1), 1.0)
        ratio = math.lerp(cfg.training.curriculum_start_ratio, cfg.training.curriculum_end_ratio, progress)
        percentile = math.lerp(cfg.training.difficulty_percentile, 95.0, progress)

        if self.scores:
            scores_t = torch.tensor(self.scores)
            threshold = scores_t.quantile(percentile / 100).item()
            hard = [i for i, s in zip(self.normal_indices, self.scores) if s >= threshold]
            if not hard:
                hard = self.normal_indices
        else:
            hard = self.normal_indices

        n_normals = min(int(len(self.attack_indices) * ratio), len(hard))
        if n_normals and n_normals < len(hard):
            perm = torch.randperm(len(hard))[:n_normals]
            selected = [hard[i] for i in perm.tolist()]
        else:
            selected = hard
        self._active_indices = selected + self.attack_indices
        self._cached_len = None

    def __iter__(self):
        perm = torch.randperm(len(self._active_indices)).tolist()
        if self.max_num_nodes is None:
            bs = max(8, self.cfg.training.batch_size)
            for start in range(0, len(perm), bs):
                yield [self._active_indices[perm[j]] for j in range(start, min(start + bs, len(perm)))]
            return
        batch, batch_nodes = [], 0
        for i in perm:
            idx = self._active_indices[i]
            n = self._node_counts[idx]
            if n > self.max_num_nodes:
                continue
            if batch_nodes + n > self.max_num_nodes:
                yield batch
                batch, batch_nodes = [idx], n
            else:
                batch.append(idx)
                batch_nodes += n
        if batch:
            yield batch

    def __len__(self) -> int:
        if self._cached_len is not None:
            return self._cached_len
        if self.max_num_nodes is None:
            bs = max(8, self.cfg.training.batch_size)
            self._cached_len = max(1, (len(self._active_indices) + bs - 1) // bs)
        else:
            total = sum(self._node_counts[i] for i in self._active_indices)
            self._cached_len = max(1, total // self.max_num_nodes)
        return self._cached_len


class CurriculumDataModule(pl.LightningDataModule):
    """Curriculum learning with persistent workers.

    Builds ONE DataLoader at init. set_epoch() on the batch_sampler controls
    which graphs are yielded each epoch — no DataLoader rebuild needed.
    """

    def __init__(self, normals, attacks, scores, val_data, cfg):
        super().__init__()
        self.val_data = val_data
        self.cfg = cfg
        self._current_epoch = 0

        full_dataset = normals + attacks
        normal_indices = list(range(len(normals)))
        attack_indices = list(range(len(normals), len(full_dataset)))

        from torch_geometric.loader import DataLoader as PyGDataLoader

        bs = max(8, cfg.training.batch_size)
        nw = cfg.num_workers
        common = dict(
            num_workers=nw, persistent_workers=nw > 0, pin_memory=True,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if cfg.training.dynamic_batching:
            info = compute_node_budget(bs, cfg)
            self._mean_nodes = info.mean_nodes
            self._batch_sampler = CurriculumDynamicBatchSampler(
                full_dataset, normal_indices, attack_indices, scores, cfg, info.budget,
            )
        else:
            self._batch_sampler = CurriculumDynamicBatchSampler(
                full_dataset, normal_indices, attack_indices, scores, cfg, max_num_nodes=None,
            )
            self._mean_nodes = None

        self._train_loader = PyGDataLoader(full_dataset, batch_sampler=self._batch_sampler, **common)

    def train_dataloader(self):
        self._batch_sampler.set_epoch(self._current_epoch)
        self._current_epoch += 1
        return self._train_loader

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader as PyGDataLoader, DynamicBatchSampler

        bs = max(8, self.cfg.training.batch_size)
        nw = self.cfg.num_workers
        common = dict(
            num_workers=nw, pin_memory=True, persistent_workers=nw > 0,
            multiprocessing_context="spawn" if nw > 0 else None,
        )

        if self.cfg.training.dynamic_batching:
            info = compute_node_budget(bs, self.cfg)
            num_steps = max(1, int(len(self.val_data) * self._mean_nodes / info.budget))
            sampler = DynamicBatchSampler(
                self.val_data, max_num=info.budget, mode="node", shuffle=False,
                num_steps=num_steps, skip_too_big=True,
            )
            return PyGDataLoader(self.val_data, batch_sampler=sampler, **common)

        return PyGDataLoader(self.val_data, batch_size=bs, shuffle=False, **common)
