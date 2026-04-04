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


# ---------------------------------------------------------------------------
# Test: Decision threshold actually used
# ---------------------------------------------------------------------------


class TestDecisionThreshold:
    """Fusion agents use decision_threshold for prediction, not hardcoded 0.5."""

    @staticmethod
    def _make_fusion_states(n: int = 32) -> torch.Tensor:
        """Create synthetic 15-D fusion state vectors."""
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        torch.manual_seed(123)
        return torch.rand(n, state_dim)

    def test_dqn_high_threshold_suppresses_positives(self):
        """With threshold=0.9, fused_scores in [0.5, 0.9) yield preds=0, not 1."""
        from graphids.core.models.fusion.dqn import DQNFusionModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        agent = DQNFusionModule(
            alpha_steps=21,
            state_dim=state_dim,
            decision_threshold=0.9,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )

        states = self._make_fusion_states()
        labels = torch.ones(len(states), dtype=torch.long)

        result = agent.validate_batch(states, labels)

        assert result["accuracy"] < 0.5, (
            f"Accuracy {result['accuracy']:.2f} is too high for threshold=0.9 on all-positive "
            f"labels — decision_threshold is likely not being used"
        )

    def test_bandit_high_threshold_suppresses_positives(self):
        """BanditFusionModule with threshold=0.9 suppresses positive predictions."""
        from graphids.core.models.fusion.bandit import BanditFusionModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        agent = BanditFusionModule(
            state_dim=state_dim,
            alpha_steps=21,
            decision_threshold=0.9,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )

        states = self._make_fusion_states()
        labels = torch.ones(len(states), dtype=torch.long)

        result = agent.validate_batch(states, labels)

        assert result["accuracy"] < 0.5, (
            f"Bandit accuracy {result['accuracy']:.2f} too high for threshold=0.9 — "
            f"decision_threshold is likely not being used"
        )

    def test_threshold_difference_changes_predictions(self):
        """Same agent state with threshold=0.1 vs 0.9 produces different predictions."""
        from graphids.core.models.fusion.dqn import DQNFusionModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        states = self._make_fusion_states()

        agent_low = DQNFusionModule(
            alpha_steps=21,
            state_dim=state_dim,
            decision_threshold=0.1,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        agent_high = DQNFusionModule(
            alpha_steps=21,
            state_dim=state_dim,
            decision_threshold=0.9,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        # Copy weights so Q-networks are identical
        agent_high.q_network.load_state_dict(agent_low.q_network.state_dict())
        agent_high.target_network.load_state_dict(agent_low.target_network.state_dict())

        result_low = agent_low.predict(states)
        result_high = agent_high.predict(states)

        # Low threshold → more positive predictions; high threshold → fewer
        assert result_low["preds"].sum() > result_high["preds"].sum(), (
            f"Low threshold should predict more positives ({result_low['preds'].sum()}) "
            f"than high threshold ({result_high['preds'].sum()})"
        )
