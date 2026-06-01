"""Stable public budget API.

The implementation lives in :mod:`graphids.core.budgeting` so the CUDA probe,
heuristic planner, dataset statistics, and worker autosizing logic can be
tested independently. Keep imports from this module stable for callers.
"""

from __future__ import annotations

from graphids.core.budgeting import (
    BudgetConfig,
    BudgetResult,
    _dataset_size_stats,
    _dataset_size_tensors,
    _heuristic_budget,
    _target_bytes,
    autosize_workers,
    collect_batch,
    probe,
)
from graphids.core.budgeting.planner import node_budget as _node_budget


def node_budget(
    dataset: str,
    *,
    model=None,
    train_dataset=None,
    conv_type: str | None = None,
    heads: int | None = None,
    min_steps: int | None = None,
) -> BudgetResult:
    """Return the node/edge packing budget for a training dataset."""
    return _node_budget(
        dataset,
        model=model,
        train_dataset=train_dataset,
        conv_type=conv_type,
        heads=heads,
        min_steps=min_steps,
        probe_fn=probe,
    )


__all__ = [
    "BudgetConfig",
    "BudgetResult",
    "_dataset_size_stats",
    "_dataset_size_tensors",
    "_heuristic_budget",
    "_target_bytes",
    "autosize_workers",
    "collect_batch",
    "node_budget",
    "probe",
]
