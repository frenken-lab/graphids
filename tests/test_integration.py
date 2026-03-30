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

    @pytest.mark.slow
    def test_gat_output_respects_num_classes(self, gat_cfg):
        """GATWithJK.from_config uses cfg.num_classes for output dim, not a hardcoded 2."""
        from graphids.core.models.gat import GATWithJK

        cfg = copy.deepcopy(gat_cfg)
        cfg.num_classes = 5

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=4)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (4, 5), (
            f"Expected output shape (4, 5) for num_classes=5, got {out.shape}"
        )

    @pytest.mark.slow
    def test_gat_output_default_binary(self, gat_cfg):
        """Default num_classes=2 produces shape [batch, 2]."""
        from graphids.core.models.gat import GATWithJK

        cfg = copy.deepcopy(gat_cfg)
        assert cfg.num_classes == 2

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=3)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (3, 2), f"Expected (3, 2), got {out.shape}"

    @pytest.mark.slow
    def test_gat_nondefault_classes(self, gat_cfg):
        """from_config() → forward() with non-default classes."""
        from graphids.core.models.gat import GATWithJK

        cfg = copy.deepcopy(gat_cfg)
        cfg.num_classes = 7

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=2)
        with torch.no_grad():
            out = model(batch)

        assert out.shape[1] == 7, f"Expected 7 output classes, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# Test: Decision threshold actually used
# ---------------------------------------------------------------------------


class TestDecisionThreshold:
    """Fusion agents use decision_threshold for prediction, not hardcoded 0.5."""

    @staticmethod
    def _make_fusion_states(n: int = 32) -> torch.Tensor:
        """Create synthetic 15-D fusion state vectors."""
        from graphids.core.models.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        torch.manual_seed(123)
        return torch.rand(n, state_dim)

    @pytest.mark.slow
    def test_dqn_high_threshold_suppresses_positives(self):
        """With threshold=0.9, fused_scores in [0.5, 0.9) yield preds=0, not 1."""
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        agent = EnhancedDQNFusionAgent(
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

    @pytest.mark.slow
    def test_bandit_high_threshold_suppresses_positives(self):
        """NeuralLinUCBAgent with threshold=0.9 suppresses positive predictions."""
        from graphids.core.models.bandit import NeuralLinUCBAgent
        from graphids.core.models.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        agent = NeuralLinUCBAgent(
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

    @pytest.mark.slow
    def test_threshold_difference_changes_predictions(self):
        """Same agent state with threshold=0.1 vs 0.9 produces different predictions."""
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        states = self._make_fusion_states()
        labels = torch.randint(0, 2, (len(states),))

        agent_low = EnhancedDQNFusionAgent(
            alpha_steps=21,
            state_dim=state_dim,
            decision_threshold=0.1,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        agent_high = EnhancedDQNFusionAgent(
            alpha_steps=21,
            state_dim=state_dim,
            decision_threshold=0.9,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        # Copy weights so Q-networks are identical
        agent_high.q_network.load_state_dict(agent_low.q_network.state_dict())
        agent_high.target_network.load_state_dict(agent_low.target_network.state_dict())

        result_low = agent_low.validate_batch(states, labels)
        result_high = agent_high.validate_batch(states, labels)

        assert result_low["accuracy"] != result_high["accuracy"], (
            "Threshold 0.1 and 0.9 produced identical accuracy — "
            "decision_threshold has no effect on predictions"
        )
