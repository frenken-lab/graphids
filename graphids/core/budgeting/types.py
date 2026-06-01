"""Shared budget result types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetResult:
    budget: int
    edge_budget: int
    binding: str = "measured"
    backward_multiplier: float = 2.0
    t_fwd: float = 0.0
    target_bytes: int = 0
