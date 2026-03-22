"""Integration tests — verify cross-component wiring, not isolated units.

Each test exercises a real code path end-to-end: config resolution → data →
model construction → inference. Imports helpers from conftest.py.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Batch, Data

from conftest import EDGE_DIM, IN_CHANNELS, NUM_IDS, N_NODES, make_batch, make_graph


# ---------------------------------------------------------------------------
# Test 1: populate_config flow
# ---------------------------------------------------------------------------


class TestPopulateConfig:
    """CANBusDataModule.populate_config writes data-derived dims into cfg."""

    @staticmethod
    def _make_datamodule_stub(graphs: list[Data]) -> object:
        """Minimal stub that satisfies populate_config's property protocol.

        We avoid constructing a real CANBusDataModule (needs filesystem data).
        Instead we subclass it and override the properties that populate_config
        reads, backed by the synthetic graphs we supply.
        """
        from graphids.core.preprocessing.datamodule import CANBusDataModule

        class StubDataModule(CANBusDataModule):
            """Bypass __init__ / setup; inject graphs directly."""

            def __init__(self, graphs: list[Data]):
                # Skip super().__init__ — we don't need hparams / filesystem
                self._graphs = graphs

            @property
            def num_ids(self) -> int:
                ids = torch.cat([g.x[:, 0] for g in self._graphs])
                return int(ids.unique().numel())

            @property
            def in_channels(self) -> int:
                return self._graphs[0].x.shape[1]

            @property
            def num_classes(self) -> int:
                labels = torch.tensor([g.y.item() for g in self._graphs])
                n = int(labels.unique().numel())
                return n if n >= 2 else 2

        return StubDataModule(graphs)

    @staticmethod
    def _base_cfg():
        from graphids.config import resolve

        return resolve("model_type=vgae", "scale=small", "lake_root=/tmp", "device=cpu")

    def test_populate_sets_default_dimensions(self):
        """populate_config writes in_channels=31, edge_dim=12, num_classes=2 from data."""
        graphs = [make_graph() for _ in range(10)]
        dm = self._make_datamodule_stub(graphs)
        cfg = self._base_cfg()

        # Pre-condition: defaults are 0 / 2
        assert cfg.in_channels == 0
        assert cfg.num_ids == 0

        dm.populate_config(cfg)

        assert cfg.in_channels == IN_CHANNELS  # 31
        assert cfg.num_ids > 0
        assert cfg.num_classes == 2  # binary labels from make_graph (y=1 only)

    def test_populate_non_default_dimensions(self):
        """populate_config reads actual data dims, not hardcoded defaults."""
        feat_dim = 25
        n_classes = 3

        def _make_custom_graph(label: int) -> Data:
            x = torch.rand(N_NODES, feat_dim)
            x[:, 0] = torch.randint(0, 5, (N_NODES,)).float()
            edge_index = torch.stack([
                torch.randint(0, N_NODES, (12,)),
                torch.randint(0, N_NODES, (12,)),
            ])
            edge_attr = torch.rand(12, 8)  # 8-D edges (non-default)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label]))

        # 3-class labels: 0, 1, 2
        graphs = [_make_custom_graph(i % n_classes) for i in range(12)]
        dm = self._make_datamodule_stub(graphs)
        cfg = self._base_cfg()

        dm.populate_config(cfg)

        assert cfg.in_channels == feat_dim, f"Expected {feat_dim}, got {cfg.in_channels}"
        assert cfg.num_classes == n_classes, f"Expected {n_classes}, got {cfg.num_classes}"

    def test_populate_single_class_floors_to_two(self):
        """If all labels are the same class, num_classes floors to 2."""
        graphs = [make_graph() for _ in range(5)]  # all y=1
        dm = self._make_datamodule_stub(graphs)
        cfg = self._base_cfg()

        dm.populate_config(cfg)

        assert cfg.num_classes == 2, "Single-class data should floor to 2"


# ---------------------------------------------------------------------------
# Test 2: FusionRewardCalculator wiring
# ---------------------------------------------------------------------------


class TestFusionRewardCalculator:
    """FusionRewardCalculator requires vgae_weights and uses constructor coefficients."""

    def test_missing_vgae_weights_raises(self):
        """vgae_weights is keyword-only and required — omitting it is a TypeError."""
        from graphids.core.models.fusion_reward import FusionRewardCalculator

        with pytest.raises(TypeError, match="vgae_weights"):
            FusionRewardCalculator()

    def test_construction_with_vgae_weights(self):
        """Providing vgae_weights succeeds and stores them."""
        from graphids.core.models.fusion_reward import FusionRewardCalculator

        weights = [0.5, 0.3, 0.2]
        calc = FusionRewardCalculator(vgae_weights=weights)

        assert torch.allclose(calc._vgae_weights, torch.tensor(weights))

    def test_reward_coefficients_actually_used(self):
        """Non-default reward coefficients produce different rewards than defaults."""
        from graphids.core.models.fusion_reward import FusionRewardCalculator
        from graphids.core.models.registry import fusion_state_dim

        state_dim = fusion_state_dim()
        n = 16
        torch.manual_seed(42)
        states = torch.rand(n, state_dim)
        labels = torch.randint(0, 2, (n,))
        alphas = torch.full((n,), 0.5)

        # Default coefficients
        calc_default = FusionRewardCalculator(vgae_weights=[0.4, 0.35, 0.25])
        norm_default = calc_default.normalize(states)
        _, gat_probs_d = calc_default.derive_scores(norm_default)
        anomaly_d, _ = calc_default.derive_scores(norm_default)
        fused_d = (1 - alphas) * anomaly_d + alphas * gat_probs_d
        preds_d = (fused_d > 0.5).long()
        rewards_default = calc_default.compute(preds_d, labels, norm_default, alphas)

        # Non-default coefficients (dramatically different)
        calc_custom = FusionRewardCalculator(
            vgae_weights=[0.1, 0.1, 0.8],
            reward_correct=10.0,
            reward_incorrect=-10.0,
            confidence_weight=2.0,
            combined_conf_weight=1.5,
            disagreement_penalty=-5.0,
            overconf_penalty=-5.0,
            balance_weight=1.0,
        )
        norm_custom = calc_custom.normalize(states)
        anomaly_c, gat_probs_c = calc_custom.derive_scores(norm_custom)
        fused_c = (1 - alphas) * anomaly_c + alphas * gat_probs_c
        preds_c = (fused_c > 0.5).long()
        rewards_custom = calc_custom.compute(preds_c, labels, norm_custom, alphas)

        # Rewards must differ — if they match, coefficients are being ignored
        assert not torch.allclose(rewards_default, rewards_custom, atol=1e-3), (
            "Default and custom reward coefficients produced identical rewards — "
            "constructor args are not being used"
        )

    def test_derive_scores_uses_vgae_weights(self):
        """derive_scores uses self._vgae_weights, not a hardcoded array."""
        from graphids.core.models.fusion_reward import FusionRewardCalculator
        from graphids.core.models.registry import fusion_state_dim

        state_dim = fusion_state_dim()
        torch.manual_seed(0)
        states = torch.rand(8, state_dim)

        calc_a = FusionRewardCalculator(vgae_weights=[1.0, 0.0, 0.0])
        calc_b = FusionRewardCalculator(vgae_weights=[0.0, 0.0, 1.0])

        scores_a, _ = calc_a.derive_scores(states)
        scores_b, _ = calc_b.derive_scores(states)

        assert not torch.allclose(scores_a, scores_b, atol=1e-4), (
            "Different vgae_weights produced identical anomaly scores — "
            "derive_scores ignores vgae_weights"
        )


# ---------------------------------------------------------------------------
# Test 3: Config → model construction flow
# ---------------------------------------------------------------------------


class TestConfigToModel:
    """resolve() → set num_classes → GATWithJK.from_config → correct output shape."""

    @pytest.mark.slow
    def test_gat_output_respects_num_classes(self, gat_cfg):
        """GATWithJK.from_config uses cfg.num_classes for output dim, not a hardcoded 2."""
        from graphids.core.models.gat import GATWithJK

        cfg = OmegaConf.create(OmegaConf.to_container(gat_cfg, resolve=True))
        with open_dict(cfg):
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

        cfg = OmegaConf.create(OmegaConf.to_container(gat_cfg, resolve=True))
        assert cfg.num_classes == 2

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=3)
        with torch.no_grad():
            out = model(batch)

        assert out.shape == (3, 2), f"Expected (3, 2), got {out.shape}"

    @pytest.mark.slow
    def test_gat_from_resolve_end_to_end(self):
        """Full path: resolve() → from_config() → forward() with non-default classes."""
        from graphids.config import resolve
        from graphids.core.models.gat import GATWithJK

        cfg = resolve("model_type=gat", "scale=small", "lake_root=/tmp", "device=cpu")
        with open_dict(cfg):
            cfg.num_classes = 7
            cfg.training.gradient_checkpointing = False

        model = GATWithJK.from_config(cfg, num_ids=NUM_IDS, in_ch=IN_CHANNELS)
        model.eval()

        batch = make_batch(n_graphs=2)
        with torch.no_grad():
            out = model(batch)

        assert out.shape[1] == 7, f"Expected 7 output classes, got {out.shape[1]}"


# ---------------------------------------------------------------------------
# Test 4: Decision threshold actually used
# ---------------------------------------------------------------------------


class TestDecisionThreshold:
    """Fusion agents use decision_threshold for prediction, not hardcoded 0.5."""

    @staticmethod
    def _make_fusion_states(n: int = 32) -> torch.Tensor:
        """Create synthetic 15-D fusion state vectors."""
        from graphids.core.models.registry import fusion_state_dim

        state_dim = fusion_state_dim()
        torch.manual_seed(123)
        return torch.rand(n, state_dim)

    @pytest.mark.slow
    def test_dqn_high_threshold_suppresses_positives(self):
        """With threshold=0.9, fused_scores in [0.5, 0.9) yield preds=0, not 1."""
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.registry import fusion_state_dim

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

        # With threshold=0.9, most fused_scores (random in ~[0,1]) will be < 0.9
        # so predictions should be predominantly 0, giving low accuracy on all-1 labels.
        # With default 0.5 threshold, ~half would be predicted 1.
        assert result["accuracy"] < 0.5, (
            f"Accuracy {result['accuracy']:.2f} is too high for threshold=0.9 on all-positive "
            f"labels — decision_threshold is likely not being used"
        )

    @pytest.mark.slow
    def test_bandit_high_threshold_suppresses_positives(self):
        """NeuralLinUCBAgent with threshold=0.9 suppresses positive predictions."""
        from graphids.core.models.bandit import NeuralLinUCBAgent
        from graphids.core.models.registry import fusion_state_dim

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
        from graphids.core.models.registry import fusion_state_dim

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
