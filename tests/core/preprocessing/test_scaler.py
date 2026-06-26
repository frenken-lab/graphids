"""Scaler module contract: strategy dispatch, benign-fit behavior, round-trip."""

from __future__ import annotations

from pathlib import Path

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


def _fixture(num_graphs: int = 6, nodes_per: int = 4, n_feat: int = 3, seed: int = 0):
    """Two-graph-class fixture: half benign (y=0), half attack (y=1).

    Attack graphs have feature mean shifted by +5σ on feature 0 — large
    enough that the benign-only filter produces a visibly different
    feature-0 mean from a hypothetical joint fit on the same rows.
    """
    g = torch.Generator().manual_seed(seed)
    n_total = num_graphs * nodes_per
    x = torch.randn((n_total, n_feat), generator=g)
    y = torch.tensor([0] * (num_graphs // 2) + [1] * (num_graphs // 2), dtype=torch.long)
    # Bias attack-graph rows on feature 0
    for gi in range(num_graphs):
        if y[gi] == 1:
            rows = slice(gi * nodes_per, (gi + 1) * nodes_per)
            x[rows, 0] += 5.0
    data = Data(x=x, y=y)
    slices = {"x": torch.arange(num_graphs + 1, dtype=torch.long) * nodes_per}
    return data, slices, torch.arange(num_graphs, dtype=torch.long)


def test_z_benign_filters_attack_rows_from_fit():
    # CONTRACT: row-selection must actually filter. If the benign filter
    # were a no-op, the fitted feature-0 mean would equal the all-rows
    # mean. Differential test against the all-rows reference, not a
    # formula mirror.
    data, slices, train_idx = _fixture()
    benign = scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
    assert set(benign["x"]) == {"mean", "std"}
    all_rows_mean_feat0 = float(data.x[:, 0].mean())
    assert abs(all_rows_mean_feat0 - float(benign["x"]["mean"][0])) > 1.0


def test_robust_benign_returns_robust_stats():
    # CONTRACT: robust strategy must emit median/IQR stats, not mean/std.
    data, slices, train_idx = _fixture()
    fitted = scaler_mod.fit(data, slices, train_idx, strategy="robust_benign", keys=("x",))
    assert set(fitted["x"]) == {"median", "iqr"}


def test_unknown_strategy_raises():
    data, slices, train_idx = _fixture()
    with pytest.raises(ValueError, match="unknown strategy"):
        scaler_mod.fit(data, slices, train_idx, strategy="not_a_real_one")


def test_torch_save_load_round_trip(tmp_path: Path):
    # CONTRACT: torch.save on a dict of tensor stats round-trips.
    data, slices, train_idx = _fixture()
    fitted = scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
    path = tmp_path / "feature_scaler.pt"
    torch.save(fitted, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    # Apply both to a fresh copy and compare
    d1 = Data(x=data.x.clone())
    d2 = Data(x=data.x.clone())
    scaler_mod.apply(d1, fitted)
    scaler_mod.apply(d2, reloaded)
    assert torch.allclose(d1.x, d2.x)


def test_apply_preserves_dtype():
    # REGRESSION: numpy round-trip in apply() must return to the original
    # tensor dtype, not numpy's default float64.
    data, slices, train_idx = _fixture()
    data.x = data.x.to(torch.float32)
    fitted = scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
    scaler_mod.apply(data, fitted)
    assert data.x.dtype == torch.float32


def test_scaler_config_helpers_round_trip():
    zcfg = ZBenignScalerCfg()
    assert scaler_kind(zcfg) == "z_benign"

    rcfg = RobustBenignScalerCfg()
    assert scaler_kind(rcfg) == "robust_benign"

    data, slices, train_idx = _fixture()
    fitted = fit_from_cfg(data, slices, train_idx, cfg=zcfg, keys=("x",))
    assert "x" in fitted
