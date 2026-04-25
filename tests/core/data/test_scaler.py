"""Scaler module contract: strategy dispatch, benign vs joint differential, round-trip.

Three regression / contract tests guarding the 2026-04-25 promotion of
scaling out of ``datasets/can_bus.py`` into ``core/data/scaler.py``.
sklearn's StandardScaler / RobustScaler internals aren't tested here —
that's sklearn's job.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch_geometric.data import Data

from graphids.core.data import scaler as scaler_mod


def _fixture(num_graphs: int = 6, nodes_per: int = 4, n_feat: int = 3, seed: int = 0):
    """Two-graph-class fixture: half benign (y=0), half attack (y=1).

    Attack graphs have feature mean shifted by +5σ on feature 0 — large
    enough that joint-fit and benign-fit produce visibly different stats.
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


def test_z_benign_differs_from_z_joint_when_attacks_skew_distribution():
    # CONTRACT: row-selection must actually filter. If z_benign and
    # z_joint produced the same fitted mean, the benign filter would be
    # dead code. Differential test, not a formula mirror.
    data, slices, train_idx = _fixture()
    joint = scaler_mod.fit(data, slices, train_idx, strategy="z_joint", keys=("x",))
    benign = scaler_mod.fit(data, slices, train_idx, strategy="z_benign", keys=("x",))
    assert isinstance(joint["x"], StandardScaler)
    # Feature-0 mean: joint sees attack-shifted rows, benign does not.
    assert abs(joint["x"].mean_[0] - benign["x"].mean_[0]) > 1.0


def test_robust_benign_returns_robust_scaler():
    # CONTRACT: STRATEGIES table dispatches to the right sklearn class.
    data, slices, train_idx = _fixture()
    fitted = scaler_mod.fit(data, slices, train_idx, strategy="robust_benign", keys=("x",))
    assert isinstance(fitted["x"], RobustScaler)


def test_unknown_strategy_raises():
    data, slices, train_idx = _fixture()
    with pytest.raises(ValueError, match="unknown strategy"):
        scaler_mod.fit(data, slices, train_idx, strategy="not_a_real_one")


def test_torch_save_load_round_trip(tmp_path: Path):
    # CONTRACT: torch.save on a dict of sklearn estimators round-trips
    # — the whole persistence layer is "no wrapper class needed".
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
