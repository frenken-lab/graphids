"""Batch samplers and DataLoader factory for graph data.

Dataset-agnostic: operates on pre-computed per-graph size tensors.
"""

from __future__ import annotations

import math

import torch


def _worker_init(worker_id: int) -> None:
    """Set file_system sharing strategy in spawn workers (not inherited from parent)."""
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")


def make_graph_loader(
    dataset, *, batch_sampler=None, batch_size=1, shuffle=False,
    num_workers: int = 0, pin_memory: bool = True,
    device: torch.device | None = None, **kwargs,
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


class NodeBudgetBatchSampler(torch.utils.data.Sampler):
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


class CurriculumSampler:
    """Curriculum selection — picks which graphs are active each epoch.

    set_epoch() updates active indices based on difficulty scores and
    epoch progress. Rebuilds an inner ``NodeBudgetBatchSampler`` each epoch
    over the active subset.

    ``dataset_sizes``: per-graph node counts for the full dataset (same order
    as ``dataset``). Precomputed once at construction time — rebuilding per
    epoch is then an O(M) tensor index, not an M-graph walk. The inner
    sampler is constructed with ``indices=active_indices`` so it yields
    positions in the full dataset, not subset-local positions.
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
        dataset_sizes: torch.Tensor,
        max_num_nodes: int | None = None,
        mean_nodes: float = 1.0,
    ):
        assert len(scores) == len(normal_indices)
        assert len(dataset_sizes) == len(dataset), (
            f"dataset_sizes length {len(dataset_sizes)} != dataset length {len(dataset)}"
        )
        self.dataset = dataset
        self.dataset_sizes = dataset_sizes.to(torch.long)
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
        active = torch.as_tensor(self._active_indices, dtype=torch.long)
        active_sizes = self.dataset_sizes[active]
        num_steps = max(1, math.ceil(len(self._active_indices) * self.mean_nodes / self.max_num_nodes))
        return NodeBudgetBatchSampler(
            active_sizes,
            max_num=self.max_num_nodes,
            shuffle=True,
            skip_too_big=True,
            num_steps=num_steps,
            indices=active,  # yields full-dataset positions, not subset-local
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
