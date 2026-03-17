"""Type contracts between pipeline and core layers (PEP 544).

These Protocols define what the pipeline layer expects from core models,
catching integration bugs at type-check time without runtime overhead.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

import torch
from torch_geometric.data import Data


@runtime_checkable
class GraphModel(Protocol):
    """Contract: what pipeline expects from any graph model."""

    def forward(self, data: Data) -> torch.Tensor: ...

    @classmethod
    def from_config(cls, cfg, num_ids: int, in_ch: int) -> GraphModel: ...


class StageMetrics(TypedDict, total=False):
    """Contract: what every stage returns in metrics.json."""

    accuracy: float
    f1_macro: float
    loss: float
    val_loss: float
    best_val_loss: float
    epoch: int
