"""Scaler module invariant."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from graphids.core.data.preprocessing import scaler as scaler_mod
from graphids.core.data.preprocessing.scaler import (
    RobustBenignScalerCfg,
    ZBenignScalerCfg,
    fit_from_cfg,
    scaler_kind,
)


def _fixture(num_graphs: int = 6, nodes_per: int = 4, n_feat: int = 3):
    g = torch.Generator().manual_seed(0)
    n_total = num_graphs * nodes_per
    x = torch.randn((n_total, n_feat), generator=g)
    y = torch.tensor([0] * (num_graphs // 2) + [1] * (num_graphs // 2), dtype=torch.long)
    for gi in range(num_graphs):
        if y[gi] == 1:
            rows = slice(gi * nodes_per, (gi + 1) * nodes_per)
            x[rows, 0] += 5.0
    data = Data(x=x, y=y)
    slices = {"x": torch.arange(num_graphs + 1, dtype=torch.long) * nodes_per}
    return data, slices, torch.arange(num_graphs, dtype=torch.long)


def test_scaler_dispatch_fits_benign_rows_and_applies_in_place():
    data, slices, train_idx = _fixture()

    z_stats = scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
    assert set(z_stats["x"]) == {"mean", "std"}
    assert abs(float(data.x[:, 0].mean()) - float(z_stats["x"]["mean"][0])) > 1.0

    data.x = data.x.to(torch.float32)
    scaler_mod.apply(data, z_stats)
    assert data.x.dtype == torch.float32

    robust_stats = scaler_mod.fit(data, slices, train_idx, strategy="robust_benign", keys=("x",))
    assert set(robust_stats["x"]) == {"median", "iqr"}
    assert scaler_kind(ZBenignScalerCfg()) == "z_benign"
    assert scaler_kind(RobustBenignScalerCfg()) == "robust_benign"
    assert "x" in fit_from_cfg(data, slices, train_idx, cfg=ZBenignScalerCfg(), keys=("x",))
    with pytest.raises(ValueError, match="unknown strategy"):
        scaler_mod.fit(data, slices, train_idx, strategy="not_a_real_one")


def test_scaler_fails_loud_without_benign_training_graphs():
    data, slices, train_idx = _fixture()
    data.y[:] = 1

    with pytest.raises(ValueError, match="no benign graphs"):
        scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
