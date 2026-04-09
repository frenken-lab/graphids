"""Batch building, samplers, and DataLoader factory for graph data.

Decomposed by granularity:
- ``collect_batch`` — one batch to a node budget (probing, extraction)
- ``make_loader`` — DataLoader with spawn/prefetch defaults (training)
- ``NodeBudgetBatchSampler`` — bin-packing sampler (training)
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch


# ---------------------------------------------------------------------------
# Primitive: one batch to a node budget
# ---------------------------------------------------------------------------


def collect_batch(dataset, target_nodes: int) -> Batch:
    """Collect graphs until reaching ``target_nodes`` total. No DataLoader overhead."""
    graphs, total = [], 0
    for g in dataset:
        graphs.append(g)
        total += g.num_nodes
        if total >= target_nodes:
            break
    return Batch.from_data_list(graphs)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------


def _worker_init(worker_id: int) -> None:
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")


def _clone_collate(x):
    """Pre-batched items need cloning — PrefetchLoader pins in-place."""
    return x.clone() if hasattr(x, "clone") else x


def make_loader(
    dataset, *, batch_sampler=None, batch_size=1, shuffle=False,
    num_workers: int = 0, pin_memory: bool = True, device: torch.device | None = None,
    **kwargs,
):
    """DataLoader with spawn/persistent_workers/PrefetchLoader defaults.

    Args:
        batch_size: ``None`` for pre-batched datasets (identity collation).
        device: wraps with PrefetchLoader for async H2D transfer.
    """
    if device is not None:
        pin_memory = False

    if num_workers > 0:
        kwargs.setdefault("persistent_workers", True)
        kwargs.setdefault("multiprocessing_context", "spawn")
        kwargs.setdefault("worker_init_fn", _worker_init)

    common = dict(num_workers=num_workers, pin_memory=pin_memory, **kwargs)

    if batch_size is None:
        from torch.utils.data import DataLoader as TorchDataLoader
        loader = TorchDataLoader(dataset, batch_size=None, shuffle=shuffle,
                                 collate_fn=_clone_collate, **common)
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


# Backward compat
make_graph_loader = make_loader


# ---------------------------------------------------------------------------
# Node-budget batch sampler
# ---------------------------------------------------------------------------


class NodeBudgetBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Bin-packing sampler: yields index batches where total nodes <= ``max_num``.

    Bucket-shuffle for low batch-to-batch size variance. Optional ``indices``
    mapping for curriculum subsets.
    """

    def __init__(
        self, sizes: torch.Tensor, max_num: int, *,
        shuffle: bool = True, num_buckets: int = 20,
        skip_too_big: bool = True, num_steps: int | None = None,
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
            idx = torch.as_tensor(indices, dtype=torch.long)
            if len(idx) != len(self.sizes):
                raise ValueError(f"indices length ({len(idx)}) != sizes length ({len(self.sizes)})")
            self._index_map: list[int] | None = idx.tolist()
        else:
            self._index_map = None

    def _bucket_shuffled(self) -> list[int]:
        sorted_idx = torch.argsort(self.sizes).tolist()
        bs = max(1, (len(self.sizes) + self.num_buckets - 1) // self.num_buckets)
        buckets = [sorted_idx[i:i + bs] for i in range(0, len(sorted_idx), bs)]
        order = torch.randperm(len(buckets)).tolist()
        out: list[int] = []
        for b in order:
            perm = torch.randperm(len(buckets[b])).tolist()
            out.extend(buckets[b][p] for p in perm)
        return out

    def _emit(self, batch: list[int]) -> list[int]:
        if self._index_map is None:
            return list(batch)
        return [self._index_map[i] for i in batch]

    def __iter__(self):
        local = self._bucket_shuffled() if self.shuffle else list(range(len(self.sizes)))
        max_steps = self.num_steps or len(self.sizes)
        batch: list[int] = []
        current, steps = 0, 0
        for i in local:
            n_i = int(self.sizes[i].item())
            if n_i > self.max_num:
                if self.skip_too_big:
                    continue
                if batch:
                    yield self._emit(batch); batch, current, steps = [], 0, steps + 1
                    if steps >= max_steps: return
                yield self._emit([i]); steps += 1
                if steps >= max_steps: return
                continue
            if current + n_i > self.max_num and batch:
                yield self._emit(batch); batch, current, steps = [], 0, steps + 1
                if steps >= max_steps: return
            batch.append(i); current += n_i
        if batch and steps < max_steps:
            yield self._emit(batch)

    def __len__(self) -> int:
        if self.num_steps is not None:
            return self.num_steps
        return max(1, (int(self.sizes.sum().item()) + self.max_num - 1) // self.max_num)
