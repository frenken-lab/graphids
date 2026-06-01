"""Budget planning internals.

Import from :mod:`graphids.core.budget` for the stable public surface.
"""

from .config import BudgetConfig
from .heuristic import _heuristic_budget, _target_bytes
from .planner import node_budget
from .probe import probe
from .stats import _dataset_size_stats, _dataset_size_tensors
from .types import BudgetResult
from .workers import autosize_workers, collect_batch

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
