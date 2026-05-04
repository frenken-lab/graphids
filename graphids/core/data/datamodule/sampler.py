"""Offline FFD packer for variable-size graphs (v2).

v1 also exported a live ``NodeBudgetBatchSampler`` class. That sampler
was unreachable from ``GraphDataModule.train_dataloader`` (always shadowed
by the prebatch branch when ``dynamic_batching=True``, and the fixed-batch
branch when ``dynamic_batching=False``) — confirmed dead in v1 and dropped
in ``graph_v2``. Only the offline packer survives.

FFD: sort graphs by size desc, place each into the first bin that fits
both budgets. ~10-20% tighter than greedy sequential. The dual budget
(node + edge) is load-bearing per ``critical-constraints.md`` — single-axis
admitted edge-heavy OOMs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from structlog import get_logger

log = get_logger(__name__)


@dataclass
class _Bin:
    indices: list[int] = field(default_factory=list)
    n_sum: int = 0
    e_sum: int = 0


def pack_offline(
    sizes: torch.Tensor,
    max_num: int,
    *,
    edge_sizes: torch.Tensor | None = None,
    max_edges: int | None = None,
) -> list[list[int]]:
    """First-fit-decreasing packing under dual node + edge budget.

    Returns list of dataset-global index lists. Single-graph oversize
    (exceeds either budget alone) is skipped with one summary warning.
    """
    if max_num <= 0:
        raise ValueError(f"max_num must be positive, got {max_num}")
    if edge_sizes is not None:
        if len(edge_sizes) != len(sizes):
            raise ValueError(
                f"edge_sizes length ({len(edge_sizes)}) != sizes length ({len(sizes)})"
            )
        if max_edges is None or max_edges <= 0:
            raise ValueError("max_edges must be a positive int when edge_sizes is given")

    sizes = sizes.to(torch.long)
    es = edge_sizes.to(torch.long) if edge_sizes is not None else None
    order = torch.argsort(sizes, descending=True).tolist()

    bins: list[_Bin] = []
    skipped = 0
    for i in order:
        n_i = int(sizes[i])
        e_i = int(es[i]) if es is not None else 0
        if n_i > max_num or (max_edges is not None and e_i > max_edges):
            skipped += 1
            continue
        for b in bins:
            if b.n_sum + n_i <= max_num and (max_edges is None or b.e_sum + e_i <= max_edges):
                b.indices.append(i)
                b.n_sum += n_i
                b.e_sum += e_i
                break
        else:
            bins.append(_Bin(indices=[i], n_sum=n_i, e_sum=e_i))

    if skipped:
        log.warning(
            "sampler_skipped_oversize",
            n_skipped=skipped,
            n_total=len(sizes),
            max_nodes=max_num,
            max_edges=max_edges,
        )
    return [b.indices for b in bins]
