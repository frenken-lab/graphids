"""Offline next-fit decreasing packer for variable-size graphs."""

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
    """Pack graph indices under node and edge budgets.

    The sorted next-fit strategy is intentionally linear after sorting. Exact
    first-fit gives slightly tighter bins, but it is quadratic on large cached
    graph datasets and can spend minutes on CPU before the first GPU step.
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
    current = _Bin()
    skipped = 0
    for i in order:
        n_i = int(sizes[i])
        e_i = int(es[i]) if es is not None else 0
        if n_i > max_num or (max_edges is not None and e_i > max_edges):
            skipped += 1
            continue
        fits_current = current.n_sum + n_i <= max_num and (
            max_edges is None or current.e_sum + e_i <= max_edges
        )
        if current.indices and not fits_current:
            bins.append(current)
            current = _Bin()
        current.indices.append(i)
        current.n_sum += n_i
        current.e_sum += e_i

    if current.indices:
        bins.append(current)

    if skipped:
        log.warning(
            "sampler_skipped_oversize",
            n_skipped=skipped,
            n_total=len(sizes),
            max_nodes=max_num,
            max_edges=max_edges,
        )
    return [b.indices for b in bins]
