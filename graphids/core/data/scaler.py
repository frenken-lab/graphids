"""Per-column feature scalers for graph node/edge tensors.

Two strategies, both fit on benign-only train rows; they differ only in
the sklearn reducer (``StandardScaler`` mean/std vs ``RobustScaler``
median/IQR). The reducers come from sklearn; only the graph-aware row
indexing is custom.

The fit-time / apply-time split matches sklearn's estimator contract:
``fit`` returns a dict of fitted estimators keyed by tensor attribute
(``"x"``, ``"edge_attr"``); ``apply`` runs ``estimator.transform`` on
each. Persistence is plain ``torch.save`` on the dict — sklearn
estimators pickle natively.

Rationale for benign-only fit: the supervised stage's task is to detect
deviations from normal, so the input coordinate system should be defined
by normal alone. A scaler fit on benign+attack rows (the removed
``z_joint`` strategy) bakes the training-attack distribution into the
input space, attenuating discriminative axes when attack variance
dominates and degrading zero-day generalization. See
``~/plans/scaler-design-supervised-ood.md`` and graphids issue #43.
"""

from __future__ import annotations

from typing import Final

import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch import Tensor
from torch_geometric.data import Data

# (graph_filter, reducer_class) — strategy resolves to a row selector and
# an sklearn estimator. Add a strategy by appending here.
STRATEGIES: Final[dict[str, tuple[str, type]]] = {
    "z_benign": ("benign", StandardScaler),
    "robust_benign": ("benign", RobustScaler),
}


def _select_graphs(data: Data, train_idx: Tensor, filter_name: str) -> Tensor:
    if filter_name == "benign":
        return train_idx[data.y[train_idx] == 0]
    raise ValueError(f"unknown filter {filter_name!r}")


def _slice_rows(cum: Tensor, graph_idx: Tensor) -> Tensor:
    """Expand graph indices into flat row indices for PyG ragged storage."""
    starts, ends = cum[graph_idx].long(), cum[graph_idx + 1].long()
    widths = ends - starts
    base = torch.repeat_interleave(starts, widths)
    offsets = torch.arange(int(widths.sum()), dtype=torch.long)
    offsets -= torch.repeat_interleave(widths.cumsum(0) - widths, widths)
    return base + offsets


def fit(
    data: Data,
    slices: dict[str, Tensor],
    train_idx: Tensor,
    *,
    strategy: str,
    keys: tuple[str, ...] = ("x", "edge_attr"),
) -> dict[str, StandardScaler | RobustScaler]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected {list(STRATEGIES)}")
    filter_name, Reducer = STRATEGIES[strategy]
    graphs = _select_graphs(data, train_idx, filter_name)
    fitted: dict = {}
    for key in keys:
        if not hasattr(data, key) or getattr(data, key) is None:
            continue
        rows = getattr(data, key).index_select(0, _slice_rows(slices[key], graphs))
        fitted[key] = Reducer().fit(rows.to(torch.float32).numpy())
    return fitted


def apply(data: Data, scalers: dict) -> None:
    for key, scaler in scalers.items():
        t = getattr(data, key)
        setattr(data, key, torch.from_numpy(scaler.transform(t.numpy())).to(t.dtype))
