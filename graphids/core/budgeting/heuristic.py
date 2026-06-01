"""Pure heuristic budget planning."""

from __future__ import annotations

import math

import torch
from structlog import get_logger

from .config import MB, BudgetConfig
from .stats import _dataset_size_stats
from .types import BudgetResult

log = get_logger(__name__)


def _target_bytes(config: BudgetConfig | None = None) -> int:
    cfg = config or BudgetConfig.from_env()
    if torch.cuda.is_available():
        try:
            free = int(torch.cuda.mem_get_info()[0])
        except Exception:  # pragma: no cover - defensive around mocked CUDA
            free = cfg.default_target_bytes
    else:
        free = cfg.default_target_bytes
    return max(1, int(free * cfg.safety_margin))


def _heuristic_budget(
    dataset: str,
    *,
    train_dataset=None,
    quadratic: bool = False,
    heads: int | None = None,
    min_steps: int | None = None,
    binding: str = "heuristic",
    config: BudgetConfig | None = None,
) -> BudgetResult:
    """Return a deterministic node/edge budget without running the model."""
    cfg = config or BudgetConfig.from_env()
    target = _target_bytes(cfg)
    max_nodes, max_edges, total_nodes, epn = _dataset_size_stats(train_dataset)
    reserve = int(target * cfg.cudnn_reserve)
    usable = max(1, target - reserve)

    if quadratic:
        head_factor = max(1.0, float(heads or 1) / 4.0)
        budget = int(math.sqrt(usable / (cfg.heuristic_gps_bytes_per_node2 * head_factor)))
    else:
        per_node = cfg.heuristic_bytes_per_node + int(epn * cfg.heuristic_bytes_per_edge)
        budget = int(usable / max(1, per_node))
    budget = max(max_nodes, budget, 1)

    if min_steps is not None and min_steps > 1 and total_nodes > 0:
        step_cap = total_nodes // min_steps
        if step_cap > max_nodes:
            budget = min(budget, step_cap)

    edge_budget = max(max_edges, int(budget * epn * cfg.edge_headroom), 1)
    log.info(
        "budget_heuristic",
        dataset=dataset,
        quadratic=quadratic,
        target_mb=target // MB,
        budget_nodes=budget,
        budget_edges=edge_budget,
        max_nodes=max_nodes,
        max_edges=max_edges,
        edges_per_node=round(epn, 2),
        binding=binding,
    )
    return BudgetResult(
        budget=budget,
        edge_budget=edge_budget,
        binding=binding,
        target_bytes=target,
    )
