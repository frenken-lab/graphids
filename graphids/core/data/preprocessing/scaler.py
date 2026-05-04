"""Per-column feature scalers (v2 — pure torch).

Two strategies, both fit on benign-only train rows:
- ``z_benign``: ``torch.std_mean`` → params {mean, std}
- ``robust_benign``: ``torch.quantile`` → params {median, iqr}

v1 routed through sklearn ``StandardScaler`` / ``RobustScaler``. The
sklearn estimators forced numpy round-trips on every fit/apply and
pickled awkwardly. ``torch.std_mean`` and ``torch.quantile`` give the
same statistics natively in tensor-land. Persisted scaler is a plain
``dict[str, dict[str, Tensor]]``; ``torch.save`` is the codec.

Strategy is implicit in which keys the params dict holds — no sentinel.

Benign-only rationale: see ``~/plans/scaler-design-supervised-ood.md``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data

STRATEGIES = ("z_benign", "robust_benign")
_EPS = 1e-12


def _flat_rows(cum: Tensor, graph_idx: Tensor) -> Tensor:
    """Cumsum offsets + graph indices → flat row indices. Pure torch:
    ``index_select`` + ``repeat_interleave`` + ``arange`` + ``cumsum``.
    """
    starts = cum[graph_idx].long()
    widths = (cum[graph_idx + 1] - cum[graph_idx]).long()
    return torch.repeat_interleave(starts, widths) + (
        torch.arange(int(widths.sum()), dtype=torch.long)
        - torch.repeat_interleave(widths.cumsum(0) - widths, widths)
    )


def fit(
    data: Data,
    slices: dict[str, Tensor],
    train_idx: Tensor,
    *,
    strategy: str,
    keys: tuple[str, ...] = ("x", "edge_attr"),
) -> dict[str, dict[str, Tensor]]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected {STRATEGIES}")
    benign = train_idx[data.y[train_idx] == 0]
    out: dict[str, dict[str, Tensor]] = {}
    for key in keys:
        t = getattr(data, key, None)
        if t is None:
            continue
        rows = t.index_select(0, _flat_rows(slices[key], benign)).float()
        if strategy == "z_benign":
            std, mean = torch.std_mean(rows, dim=0, unbiased=False)
            out[key] = {"mean": mean, "std": std.clamp_min(_EPS)}
        else:
            q = torch.quantile(rows, torch.tensor([0.25, 0.5, 0.75]), dim=0)
            out[key] = {"median": q[1], "iqr": (q[2] - q[0]).clamp_min(_EPS)}
    return out


def apply(data: Data, scalers: dict[str, dict[str, Tensor]]) -> None:
    for key, p in scalers.items():
        t = getattr(data, key)
        if "mean" in p:
            scaled = (t.float() - p["mean"]) / p["std"]
        else:
            scaled = (t.float() - p["median"]) / p["iqr"]
        setattr(data, key, scaled.to(t.dtype))
