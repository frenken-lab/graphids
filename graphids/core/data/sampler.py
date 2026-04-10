"""Node-budget batch sampler for variable-size graphs.

Bin-packing sampler that yields index batches where total nodes <= a budget.
Bucket-shuffle for low batch-to-batch size variance.
"""

from __future__ import annotations

import torch


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
