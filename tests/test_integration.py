"""Integration tests — verify cross-component wiring, not isolated units.

Each test exercises a real code path end-to-end: config → model construction →
inference. Imports helpers from conftest.py.
"""

from __future__ import annotations

import copy

import pytest
import torch
from conftest import IN_CHANNELS, NUM_IDS, make_batch

# ---------------------------------------------------------------------------
# Test: Config → model construction flow
# ---------------------------------------------------------------------------


class TestConfigToModel:
    """Config → GATWithJK.from_config → correct output shape."""

    @pytest.mark.parametrize("num_classes, n_graphs", [
        (2, 3),
        (5, 4),
        (7, 2),
    ], ids=["default_binary", "five_class", "seven_class"])
    def test_gat_output_shape_matches_num_classes(self, gat_cfg, num_classes, n_graphs):
        from graphids.core.models.supervised.gat import GATWithJK

        cfg = copy.deepcopy(gat_cfg)
        cfg.num_classes = num_classes

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=n_graphs)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (n_graphs, num_classes), (
            f"Expected ({n_graphs}, {num_classes}), got {out.shape}"
        )

