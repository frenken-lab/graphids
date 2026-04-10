"""Fusion model tests: MLP/WeightedAvg/RL modules, reward calculator, checkpoints."""

from __future__ import annotations

import pytest
import torch


class TestFusionBaselineTestStep:
    """MLPFusionModule and WeightedAvgModule must have test_step + test_metrics."""

    @staticmethod
    def _make_fusion_batch(n: int = 32):
        from graphids.core.models.fusion.fusion_features import STATE_DIM

        sd = STATE_DIM
        states = torch.rand(n, sd)
        labels = torch.randint(0, 2, (n,))
        return states, labels

    @staticmethod
    def _make_module(name):
        from graphids.core.models.fusion.fusion_features import STATE_DIM

        from graphids.core.models.fusion.mlp import MLPFusionModule
        from graphids.core.models.fusion.weighted_avg import WeightedAvgModule

        if name == "mlp":
            return MLPFusionModule(state_dim=STATE_DIM)
        return WeightedAvgModule()

    @pytest.mark.parametrize("module_name", ["mlp", "weighted_avg"])
    def test_has_test_metrics(self, module_name):
        module = self._make_module(module_name)
        assert hasattr(module, "test_metrics"), f"{type(module).__name__} missing test_metrics"

    @pytest.mark.parametrize("module_name", ["mlp", "weighted_avg"])
    def test_test_step_updates_metrics(self, module_name):
        module = self._make_module(module_name)
        module.eval()
        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(), 0)
        result = module.test_metrics.compute()
        assert "accuracy" in result
        assert "f1" in result
        assert all(0.0 <= v.item() <= 1.0 for v in result.values())

    @pytest.mark.parametrize("module_name", ["mlp", "weighted_avg"])
    def test_metrics_reset_between_scenarios(self, module_name):
        module = self._make_module(module_name)
        module.eval()
        module.on_test_epoch_start()
        module.test_step(self._make_fusion_batch(32), 0)
        result_1 = module.test_metrics.compute()
        module.test_metrics.reset()
        module.test_step(self._make_fusion_batch(16), 0)
        result_2 = module.test_metrics.compute()
        assert "accuracy" in result_1
        assert "accuracy" in result_2


class TestFusionRewardCalculator:
    """FusionRewardCalculator requires vgae_weights + paper reward constants.

    Reward constants live in ``configs/fusion/reward.libsonnet`` and are
    passed through by the method libsonnets; missing constants fall back
    to the libsonnet (no hardcoded defaults).
    """

    # Paper values from configs/fusion/reward.libsonnet — kept in sync by
    # ``test_reward_libsonnet_matches_paper_values`` below.
    _PAPER_REWARD = {
        "correct": 3.0,
        "incorrect": -3.0,
        "confidence_weight": 0.5,
        "combined_conf_weight": 0.3,
        "disagreement_penalty": -1.0,
        "overconf_penalty": -1.5,
        "balance_weight": 0.3,
    }

    def test_missing_vgae_weights_raises(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator

        with pytest.raises(TypeError, match="vgae_weights"):
            FusionRewardCalculator(**self._PAPER_REWARD)

    def test_construction_with_vgae_weights(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator

        weights = [0.5, 0.3, 0.2]
        calc = FusionRewardCalculator(vgae_weights=weights, **self._PAPER_REWARD)
        assert torch.allclose(calc._vgae_weights, torch.tensor(weights))

    def test_derive_scores_uses_vgae_weights(self):
        from graphids.core.models.fusion.fusion_features import STATE_DIM
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator

        state_dim = STATE_DIM
        torch.manual_seed(0)
        states = torch.rand(8, state_dim)

        calc_a = FusionRewardCalculator(vgae_weights=[1.0, 0.0, 0.0], **self._PAPER_REWARD)
        calc_b = FusionRewardCalculator(vgae_weights=[0.0, 0.0, 1.0], **self._PAPER_REWARD)

        scores_a, _ = calc_a.derive_scores(states)
        scores_b, _ = calc_b.derive_scores(states)

        assert not torch.allclose(scores_a, scores_b, atol=1e-4)

    def test_reward_libsonnet_matches_paper_values(self):
        """configs/fusion/reward.libsonnet is the single source of truth —
        this test catches drift if someone edits the libsonnet without
        updating the _PAPER_REWARD fixture (or vice versa)."""
        import json
        import shutil
        import subprocess

        from graphids.config.constants import PROJECT_ROOT

        jsonnet_bin = shutil.which("jsonnet")
        if not jsonnet_bin:
            pytest.skip("jsonnet binary not on PATH")
        reward_path = PROJECT_ROOT / "configs" / "fusion" / "reward.libsonnet"
        result = subprocess.run(
            [jsonnet_bin, str(reward_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        rendered = json.loads(result.stdout)
        assert rendered == self._PAPER_REWARD


class TestFusionCheckpointRoundtrip:
    """Fusion checkpoint save/load format consistency."""

    def test_mlp_roundtrip(self, tmp_path):
        from graphids.core.models.fusion.fusion_features import STATE_DIM

        from graphids.core.models.fusion.mlp import MLPFusionModule

        m1 = MLPFusionModule(state_dim=STATE_DIM)
        m1.eval()
        ckpt_path = tmp_path / "mlp.ckpt"
        torch.save({"state_dict": m1.state_dict()}, ckpt_path)

        m2 = MLPFusionModule(state_dim=STATE_DIM)
        m2.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"])
        m2.eval()
        x = torch.rand(8, STATE_DIM)
        with torch.no_grad():
            torch.testing.assert_close(m1(x), m2(x))

    def test_weighted_avg_roundtrip(self, tmp_path):
        from graphids.core.models.fusion.fusion_features import STATE_DIM

        from graphids.core.models.fusion.weighted_avg import WeightedAvgModule

        m1 = WeightedAvgModule()
        m1.weight.data.fill_(0.7)
        m1.eval()
        ckpt_path = tmp_path / "wavg.ckpt"
        torch.save({"state_dict": m1.state_dict()}, ckpt_path)

        m2 = WeightedAvgModule()
        m2.load_state_dict(torch.load(ckpt_path, weights_only=True)["state_dict"])
        m2.eval()
        x = torch.rand(8, STATE_DIM)
        with torch.no_grad():
            torch.testing.assert_close(m1(x), m2(x))

    def test_dqn_roundtrip(self, tmp_path):
        from graphids.core.models.fusion.fusion_features import STATE_DIM

        from graphids.core.models.fusion.dqn import DQNFusionModule

        sd = STATE_DIM
        a1 = DQNFusionModule(
            alpha_steps=11,
            state_dim=sd,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        ckpt = {
            "q_network": a1.q_network.state_dict(),
            "epsilon": a1.epsilon,
        }
        torch.save(ckpt, tmp_path / "dqn.pt")
        a2 = DQNFusionModule(
            alpha_steps=11,
            state_dim=sd,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        ckpt_loaded = torch.load(tmp_path / "dqn.pt", weights_only=True)
        a2.q_network.load_state_dict(ckpt_loaded["q_network"])
        a2.epsilon = ckpt_loaded["epsilon"]
        a1.q_network.eval()
        a2.q_network.eval()
        x = torch.rand(8, sd)
        with torch.no_grad():
            q1 = a1.q_network(x)
            q2 = a2.q_network(x)
        torch.testing.assert_close(q1, q2)
        assert a1.epsilon == a2.epsilon
