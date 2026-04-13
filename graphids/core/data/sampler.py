"""Node-budget batch sampler for variable-size graphs.

Bin-packing sampler that yields index batches honoring a node budget, and
optionally an edge budget as a dual constraint. The dual constraint matters
when per-batch memory is dominated by message-passing activations (∝ edges)
rather than node features (∝ nodes) — the edge budget prevents rare
dense-edge graphs from OOMing even when the node budget would admit them.
"""

from __future__ import annotations

import torch

from graphids._otel import get_logger

log = get_logger(__name__)


class NodeBudgetBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Bin-packing sampler with optional dual node/edge budget.

    - ``sizes`` / ``max_num``: per-graph node counts, max nodes per batch.
    - ``edge_sizes`` / ``max_edges`` (optional): per-graph edge counts, max
      edges per batch. A batch closes when adding a graph would exceed
      EITHER budget. A graph exceeding either budget on its own is oversize
      (skipped when ``skip_too_big=True``, else yielded alone).

    Bucket-shuffle keeps batch-to-batch size variance low. ``indices``
    maps local positions to dataset-global indices (for curriculum subsets).
    """

    def __init__(
        self, sizes: torch.Tensor, max_num: int, *,
        edge_sizes: torch.Tensor | None = None, max_edges: int | None = None,
        shuffle: bool = True, num_buckets: int = 20,
        skip_too_big: bool = True,
        indices: torch.Tensor | list[int] | None = None,
    ):
        if max_num <= 0:
            raise ValueError(f"max_num must be positive, got {max_num}")
        self.sizes = sizes.to(torch.long)
        self.max_num = int(max_num)

        if edge_sizes is not None:
            if len(edge_sizes) != len(self.sizes):
                raise ValueError(
                    f"edge_sizes length ({len(edge_sizes)}) "
                    f"!= sizes length ({len(self.sizes)})"
                )
            if max_edges is None or max_edges <= 0:
                raise ValueError(
                    "max_edges must be a positive int when edge_sizes is given"
                )
            self.edge_sizes: torch.Tensor | None = edge_sizes.to(torch.long)
            self.max_edges: int | None = int(max_edges)
        else:
            self.edge_sizes = None
            self.max_edges = None

        self.shuffle = shuffle
        self.num_buckets = max(1, int(num_buckets))
        self.skip_too_big = skip_too_big
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

    def _oversize(self, n_i: int, e_i: int) -> bool:
        if n_i > self.max_num:
            return True
        if self.max_edges is not None and e_i > self.max_edges:
            return True
        return False

    def _would_exceed(self, cur_n: int, cur_e: int, n_i: int, e_i: int) -> bool:
        if cur_n + n_i > self.max_num:
            return True
        if self.max_edges is not None and cur_e + e_i > self.max_edges:
            return True
        return False

    def __iter__(self):
        local = self._bucket_shuffled() if self.shuffle else list(range(len(self.sizes)))
        has_edges = self.edge_sizes is not None
        skipped = 0
        batch: list[int] = []
        cur_n, cur_e = 0, 0
        warn = not getattr(self, "_mute_warn", False)
        for i in local:
            n_i = int(self.sizes[i].item())
            e_i = int(self.edge_sizes[i].item()) if has_edges else 0
            if self._oversize(n_i, e_i):
                if self.skip_too_big:
                    skipped += 1
                    continue
                if batch:
                    yield self._emit(batch); batch, cur_n, cur_e = [], 0, 0
                yield self._emit([i])
                continue
            if self._would_exceed(cur_n, cur_e, n_i, e_i) and batch:
                yield self._emit(batch); batch, cur_n, cur_e = [], 0, 0
            batch.append(i)
            cur_n += n_i
            cur_e += e_i
        if batch:
            yield self._emit(batch)
        if skipped and warn:
            # One summary line per epoch, not per-graph. A non-zero count
            # here is a real signal — either the dataset has outlier giants
            # or the probe budget is too tight; either way, coverage
            # shrinks silently otherwise. Muted when __len__ probes.
            log.warning(
                "sampler_skipped_oversize",
                n_skipped=skipped, n_total=len(self.sizes),
                max_nodes=self.max_num, max_edges=self.max_edges,
            )

    def __len__(self) -> int:
        self._mute_warn = True
        try:
            return sum(1 for _ in self.__iter__())
        finally:
            self._mute_warn = False


def pack_offline(
    sizes: torch.Tensor, max_num: int, *,
    edge_sizes: torch.Tensor | None = None, max_edges: int | None = None,
    skip_too_big: bool = True,
) -> list[list[int]]:
    """First-fit-decreasing packing for the prebatch path.

    The sampler's live packing walks indices sequentially (or bucket-shuffled)
    and closes a batch greedily — ~11/9 × OPT at best, and significantly
    worse when dataset order isn't size-sorted. FFD sorts graphs by size
    descending, then places each into the first batch it fits. For variable-
    size graphs this gives ~10-20% better node-budget utilization than
    sequential packing with no epoch-to-epoch randomness to preserve.

    Returns a list of batch index lists (dataset-global indices; no
    shuffle). Used by ``GraphDataModule._prebatch`` — the class sampler
    is still used for live training where ``shuffle=True`` re-buckets
    per epoch.
    """
    if max_num <= 0:
        raise ValueError(f"max_num must be positive, got {max_num}")
    has_edges = edge_sizes is not None
    if has_edges:
        if len(edge_sizes) != len(sizes):
            raise ValueError(
                f"edge_sizes length ({len(edge_sizes)}) != sizes length ({len(sizes)})"
            )
        if max_edges is None or max_edges <= 0:
            raise ValueError("max_edges must be a positive int when edge_sizes is given")

    sizes_l = sizes.to(torch.long)
    edges_l = edge_sizes.to(torch.long) if has_edges else None
    order = torch.argsort(sizes_l, descending=True).tolist()

    # bins: list of [indices, node_sum, edge_sum]
    bins: list[list] = []
    skipped = 0
    for i in order:
        n_i = int(sizes_l[i].item())
        e_i = int(edges_l[i].item()) if has_edges else 0
        oversize = n_i > max_num or (max_edges is not None and e_i > max_edges)
        if oversize:
            if skip_too_big:
                skipped += 1
                continue
            bins.append([[i], n_i, e_i])
            continue
        placed = False
        for b in bins:
            if b[1] + n_i <= max_num and (
                max_edges is None or b[2] + e_i <= max_edges
            ):
                b[0].append(i); b[1] += n_i; b[2] += e_i
                placed = True
                break
        if not placed:
            bins.append([[i], n_i, e_i])

    if skipped:
        log.warning(
            "sampler_skipped_oversize",
            n_skipped=skipped, n_total=len(sizes_l),
            max_nodes=max_num, max_edges=max_edges,
        )
    return [b[0] for b in bins]
