"""Batch samplers and DataLoader factory for graph data.

Follows PyTorch sampler grammar:

- ``Sampler[int]`` yields individual indices
- ``Sampler[list[int]]`` yields batches (``NodeBudgetBatchSampler``)
- Compose: ``NodeBudgetBatchSampler`` wraps any index source

``make_graph_loader`` supports ``batch_size=None`` for pre-batched datasets
where each item is already a collated ``Batch`` (identity collation via
``torch.utils.data.DataLoader`` — PyG DataLoader force-overrides ``collate_fn``).

Dataset-agnostic: operates on pre-computed per-graph size tensors.
"""

from __future__ import annotations

import torch


def _worker_init(worker_id: int) -> None:
    """Set file_system sharing strategy in spawn workers (not inherited from parent)."""
    import torch.multiprocessing as mp

    mp.set_sharing_strategy("file_system")


def make_graph_loader(
    dataset,
    *,
    batch_sampler=None,
    batch_size=1,
    shuffle=False,
    num_workers: int = 0,
    pin_memory: bool = True,
    device: torch.device | None = None,
    **kwargs,
):
    """Thin wrapper around PyG DataLoader — sets spawn/persistent_workers defaults.

    Args:
        batch_size: Pass ``None`` for pre-batched datasets where each
            ``__getitem__`` already returns a complete ``Batch``.  Uses
            ``torch.utils.data.DataLoader`` with identity collation
            (PyG DataLoader force-overrides ``collate_fn``).
        device: When set, wraps the loader with PyG's PrefetchLoader for async
            H2D transfer via CUDA streams. pin_memory is disabled on the inner
            loader (PrefetchLoader handles pinning internally).
    """
    if device is not None:
        pin_memory = False  # PrefetchLoader pins internally

    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("worker_init_fn", _worker_init)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    if batch_size is None:
        # Pre-batched: each __getitem__ returns a complete Batch.
        # PyG DataLoader pops collate_fn, so use torch DataLoader directly.
        from torch.utils.data import DataLoader as TorchDataLoader

        loader = TorchDataLoader(
            dataset,
            batch_size=None,
            shuffle=shuffle,
            collate_fn=_identity,
            **common,
        )
    elif batch_sampler is not None:
        from torch_geometric.loader import DataLoader as PyGDataLoader

        loader = PyGDataLoader(dataset, batch_sampler=batch_sampler, **common)
    else:
        from torch_geometric.loader import DataLoader as PyGDataLoader

        loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)

    if device is not None:
        from torch_geometric.loader import PrefetchLoader

        return PrefetchLoader(loader, device=device)
    return loader


# ---------------------------------------------------------------------------
# Sampler[list[int]] — node-budget batch packing
# ---------------------------------------------------------------------------


class NodeBudgetBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Node-budget batch sampler that reads per-graph sizes from a precomputed tensor.

    Replaces PyG's DynamicBatchSampler, which walks ``dataset[i].num_nodes`` per
    graph per epoch — 50K mmap'd Data reconstructions per epoch on set_02.
    This sampler reads ``num_nodes_per_graph`` (derived from cache slice offsets
    at zero I/O cost) once and yields batches without ever touching the dataset.

    When ``shuffle=True``, uses bucket shuffle: sort split indices by size,
    chunk into ``num_buckets`` groups, shuffle bucket order + within-bucket
    order. Keeps batch-to-batch size variance low (reduces VRAM fragmentation)
    and — combined with the presorted v8.0.0 cache — produces sequential
    mmap page faults instead of random ones within each bucket.

    ``indices``: optional mapping from internal positions to dataset-level
    positions. When given, ``sizes`` describes a subset and the sampler yields
    ``indices[i]`` instead of ``i``. Required for curriculum learning, which
    trains on a difficulty-filtered subset that changes per-epoch but must
    yield positions in the full dataset.
    """

    def __init__(
        self,
        sizes: torch.Tensor,
        max_num: int,
        *,
        shuffle: bool = True,
        num_buckets: int = 20,
        skip_too_big: bool = True,
        num_steps: int | None = None,
        indices: torch.Tensor | list[int] | None = None,
    ):
        if max_num <= 0:
            raise ValueError(f"max_num must be positive, got {max_num}")
        self.sizes = sizes.to(torch.long)
        self.max_num = int(max_num)
        self.shuffle = shuffle
        self.num_buckets = max(1, int(num_buckets))
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        if indices is not None:
            idx_tensor = torch.as_tensor(indices, dtype=torch.long)
            if len(idx_tensor) != len(self.sizes):
                raise ValueError(
                    f"indices length ({len(idx_tensor)}) must match sizes length "
                    f"({len(self.sizes)})"
                )
            self._index_map: list[int] | None = idx_tensor.tolist()
        else:
            self._index_map = None

    def _bucket_shuffled_indices(self) -> list[int]:
        n = len(self.sizes)
        sorted_idx = torch.argsort(self.sizes).tolist()
        bucket_size = max(1, (n + self.num_buckets - 1) // self.num_buckets)
        buckets = [sorted_idx[i : i + bucket_size] for i in range(0, n, bucket_size)]
        bucket_order = torch.randperm(len(buckets)).tolist()
        out: list[int] = []
        for b_idx in bucket_order:
            bucket = buckets[b_idx]
            perm = torch.randperm(len(bucket)).tolist()
            out.extend(bucket[p] for p in perm)
        return out

    def _emit(self, local_batch: list[int]) -> list[int]:
        """Translate local (subset) positions to dataset positions if needed."""
        if self._index_map is None:
            return list(local_batch)
        return [self._index_map[i] for i in local_batch]

    def __iter__(self):
        local = self._bucket_shuffled_indices() if self.shuffle else list(range(len(self.sizes)))
        max_steps = self.num_steps or len(self.sizes)
        batch: list[int] = []
        current = 0
        steps = 0
        for i in local:
            n_i = int(self.sizes[i].item())
            if n_i > self.max_num:
                if self.skip_too_big:
                    continue
                if batch:
                    yield self._emit(batch)
                    batch, current = [], 0
                    steps += 1
                    if steps >= max_steps:
                        return
                yield self._emit([i])
                steps += 1
                if steps >= max_steps:
                    return
                continue
            if current + n_i > self.max_num and batch:
                yield self._emit(batch)
                batch, current = [], 0
                steps += 1
                if steps >= max_steps:
                    return
            batch.append(i)
            current += n_i
        if batch and steps < max_steps:
            yield self._emit(batch)

    def __len__(self) -> int:
        if self.num_steps is not None:
            return self.num_steps
        total = int(self.sizes.sum().item())
        return max(1, (total + self.max_num - 1) // self.max_num)


# ---------------------------------------------------------------------------
# Factory — VGAE scoring + difficulty-tier bucketing
# ---------------------------------------------------------------------------


def build_curriculum_tiers(
    train_ds,
    *,
    vgae_ckpt_path: str,
    canid_weight: float,
    num_tiers: int = 10,
    seed: int = 0,
) -> tuple[torch.Tensor, list[list[int]], list[int], list, torch.Tensor]:
    """Score graphs by VGAE difficulty and bucket normals into difficulty tiers.

    Loads the VGAE checkpoint on CPU, scores normal-class training graphs,
    sorts by ascending difficulty, and partitions into ``num_tiers`` quantile
    buckets. Tier 0 = easiest, tier K-1 = hardest. Attacks get a separate
    always-active tier.

    Returns ``(scores, normal_tier_indices, attack_indices, full_dataset,
    dataset_sizes)`` where ``full_dataset`` is reordered ``[normals + attacks]``
    and each tier in ``normal_tier_indices`` is a list of indices into
    ``full_dataset``.
    """
    import gc
    import math
    from pathlib import Path

    from graphids.core.models._training import load_inner_model

    if not vgae_ckpt_path:
        raise ValueError(
            "sampler='curriculum' requires vgae_ckpt_path — train a VGAE autoencoder first"
        )

    device = torch.device("cpu")
    vgae, _ = load_inner_model("vgae", Path(vgae_ckpt_path), device)

    normals = [g for g in train_ds if int(g.y[0]) == 0]
    attacks = [g for g in train_ds if int(g.y[0]) == 1]
    scores = vgae.score_difficulty(normals, canid_weight=canid_weight)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float)
    del vgae
    gc.collect()

    full_dataset = normals + attacks
    dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)

    # Sort normal indices by ascending difficulty score
    sorted_order = torch.argsort(scores).tolist()
    # Partition into num_tiers quantile buckets
    bucket_size = max(1, math.ceil(len(sorted_order) / num_tiers))
    normal_tier_indices: list[list[int]] = []
    for start in range(0, len(sorted_order), bucket_size):
        tier = sorted_order[start : start + bucket_size]
        normal_tier_indices.append(tier)

    attack_indices = list(range(len(normals), len(full_dataset)))

    return scores, normal_tier_indices, attack_indices, full_dataset, dataset_sizes


# ---------------------------------------------------------------------------
# Lightning callback — selects active curriculum tiers each epoch
# ---------------------------------------------------------------------------

import pytorch_lightning as pl


class CurriculumEpochCallback(pl.Callback):
    """Select active curriculum tiers each training epoch.

    Reads tier batches from the datamodule and updates the active batch
    list based on curriculum progression.  No-op when the datamodule
    doesn't use tier-based curriculum.  Safe to add unconditionally to
    the forced callback list.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        if getattr(dm, "_tier_batches", None) is not None:
            dm._select_active_tiers(trainer.current_epoch)


def _identity(x):
    """Identity collate_fn for ``make_graph_loader(batch_size=None)``."""
    return x
