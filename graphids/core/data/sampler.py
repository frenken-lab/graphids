"""Batch samplers and DataLoader factory for graph data.

Follows PyTorch sampler grammar:

- ``Sampler[int]`` yields individual indices (``CurriculumSampler``)
- ``Sampler[list[int]]`` yields batches (``NodeBudgetBatchSampler``)
- Compose: ``NodeBudgetBatchSampler`` wraps any index source

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
        device: When set, wraps the loader with PyG's PrefetchLoader for async
            H2D transfer via CUDA streams. pin_memory is disabled on the inner
            loader (PrefetchLoader handles pinning internally).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    if device is not None:
        pin_memory = False  # PrefetchLoader pins internally

    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("worker_init_fn", _worker_init)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    if batch_sampler is not None:
        loader = PyGDataLoader(dataset, batch_sampler=batch_sampler, **common)
    else:
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
# Sampler[int] — curriculum epoch-based subset selection
# ---------------------------------------------------------------------------


class CurriculumSampler(torch.utils.data.Sampler[int]):
    """Epoch-based difficulty-gated subset sampler (DistributedSampler pattern).

    Each epoch, ``set_epoch()`` recomputes which graph indices are active
    based on difficulty scores and a curriculum ramp. ``__iter__`` yields
    individual indices (shuffled) — pair with ``NodeBudgetBatchSampler``
    for node-budget batching.

    Follows the same ``set_epoch`` / ``__iter__`` / ``__len__`` contract
    as ``torch.utils.data.DistributedSampler``.
    """

    def __init__(
        self,
        normal_indices: list[int],
        attack_indices: list[int],
        scores: list[float] | torch.Tensor,
        *,
        max_epochs: int,
        curriculum_start_ratio: float,
        curriculum_end_ratio: float,
        difficulty_percentile: float,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.normal_indices = torch.tensor(normal_indices, dtype=torch.long)
        self.attack_indices = list(attack_indices)
        self.scores = torch.tensor(scores) if not isinstance(scores, torch.Tensor) else scores
        self.max_epochs = max_epochs
        self.curriculum_start_ratio = curriculum_start_ratio
        self.curriculum_end_ratio = curriculum_end_ratio
        self.difficulty_percentile = difficulty_percentile
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        # Initial active set: all indices
        self._active_indices: list[int] = normal_indices + attack_indices

    def set_epoch(self, epoch: int) -> None:
        """Update active indices based on difficulty scores and curriculum progress.

        Mirrors ``DistributedSampler.set_epoch`` — call before each epoch
        to change the subset and shuffle seed.
        """
        self.epoch = epoch
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

    def __iter__(self):
        indices = list(self._active_indices)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        return iter(indices)

    def __len__(self) -> int:
        return len(self._active_indices)


# ---------------------------------------------------------------------------
# Factory — VGAE scoring + sampler construction
# ---------------------------------------------------------------------------


def build_curriculum_sampler(
    train_ds,
    *,
    vgae_ckpt_path: str,
    max_epochs: int,
    curriculum_start_ratio: float,
    curriculum_end_ratio: float,
    difficulty_percentile: float,
    canid_weight: float,
    seed: int = 0,
) -> tuple[CurriculumSampler, list, torch.Tensor]:
    """Score graphs by VGAE difficulty and build a curriculum sampler.

    Loads the VGAE checkpoint on CPU, scores normal-class training graphs,
    then discards the model. Returns ``(sampler, full_dataset, dataset_sizes)``
    where ``full_dataset`` is the reordered list (normals + attacks) that the
    sampler indexes into, and ``dataset_sizes`` is the per-graph node count
    tensor for ``NodeBudgetBatchSampler``.
    """
    import gc
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
    del vgae
    gc.collect()

    full_dataset = normals + attacks
    normal_indices = list(range(len(normals)))
    attack_indices = list(range(len(normals), len(full_dataset)))
    dataset_sizes = torch.tensor([g.num_nodes for g in full_dataset], dtype=torch.long)

    sampler = CurriculumSampler(
        normal_indices,
        attack_indices,
        scores,
        max_epochs=max_epochs,
        curriculum_start_ratio=curriculum_start_ratio,
        curriculum_end_ratio=curriculum_end_ratio,
        difficulty_percentile=difficulty_percentile,
        seed=seed,
    )
    return sampler, full_dataset, dataset_sizes


# ---------------------------------------------------------------------------
# Lightning callback — bridges DataModule ↔ Sampler epoch sync
# ---------------------------------------------------------------------------

import pytorch_lightning as pl


class CurriculumEpochCallback(pl.Callback):
    """Advance curriculum sampler's epoch counter each training epoch.

    Lightning's ``LightningDataModule`` does not have an
    ``on_train_epoch_start`` hook — only ``pl.Callback`` does. This
    callback reads the datamodule's ``_curriculum_sampler`` and calls
    ``set_epoch()``. It's a no-op when the datamodule doesn't use
    curriculum sampling, so it's safe to add unconditionally to the
    forced callback list.
    """

    def on_train_epoch_start(self, trainer, pl_module):
        dm = trainer.datamodule
        sampler = getattr(dm, "_curriculum_sampler", None)
        if sampler is not None:
            sampler.set_epoch(trainer.current_epoch)
