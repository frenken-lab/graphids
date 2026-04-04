"""Fusion model tests: MLP/WeightedAvg/RL modules, reward calculator, checkpoints."""

from __future__ import annotations

import pytest
import torch


class TestFusionBaselineTestStep:
    """MLPFusionModule and WeightedAvgModule must have test_step + test_metrics."""

    @staticmethod
    def _make_fusion_batch(n: int = 32):
        from graphids.core.models.fusion.fusion_features import fusion_state_dim
        sd = fusion_state_dim()
        states = torch.rand(n, sd)
        labels = torch.randint(0, 2, (n,))
        return states, labels

    @staticmethod
    def _make_module(name):
        from graphids.core.models.fusion.fusion_baselines import MLPFusionModule, WeightedAvgModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim
        if name == "mlp":
            return MLPFusionModule(state_dim=fusion_state_dim())
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

    @pytest.mark.slow
    @pytest.mark.parametrize("module_name", ["mlp", "weighted_avg"])
    def test_test_step_via_lightning_trainer(self, module_name):
        import pytorch_lightning as pl
        from torch.utils.data import TensorDataset  # noqa: test-only tensor wrapping
        module = self._make_module(module_name)
        states, labels = self._make_fusion_batch(64)
        loader = torch.utils.data.DataLoader(TensorDataset(states, labels), batch_size=16)
        trainer = pl.Trainer(accelerator="cpu", logger=False, enable_checkpointing=False, enable_progress_bar=False)
        trainer.test(module, dataloaders=loader, verbose=False)
        result = module.test_metrics.compute()
        assert "accuracy" in result

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
    """FusionRewardCalculator requires vgae_weights and uses constructor coefficients."""

    def test_missing_vgae_weights_raises(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator
        with pytest.raises(TypeError, match="vgae_weights"):
            FusionRewardCalculator()

    def test_construction_with_vgae_weights(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator
        weights = [0.5, 0.3, 0.2]
        calc = FusionRewardCalculator(vgae_weights=weights)
        assert torch.allclose(calc._vgae_weights, torch.tensor(weights))

    def test_reward_coefficients_actually_used(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        n = 16
        torch.manual_seed(42)
        states = torch.rand(n, state_dim)
        labels = torch.randint(0, 2, (n,))
        alphas = torch.full((n,), 0.5)

        calc_default = FusionRewardCalculator(vgae_weights=[0.4, 0.35, 0.25])
        norm_default = calc_default.normalize(states)
        _, gat_probs_d = calc_default.derive_scores(norm_default)
        anomaly_d, _ = calc_default.derive_scores(norm_default)
        fused_d = (1 - alphas) * anomaly_d + alphas * gat_probs_d
        preds_d = (fused_d > 0.5).long()
        rewards_default = calc_default.compute(preds_d, labels, norm_default, alphas)

        calc_custom = FusionRewardCalculator(
            vgae_weights=[0.1, 0.1, 0.8],
            reward_correct=10.0, reward_incorrect=-10.0,
            confidence_weight=2.0, combined_conf_weight=1.5,
            disagreement_penalty=-5.0, overconf_penalty=-5.0, balance_weight=1.0,
        )
        norm_custom = calc_custom.normalize(states)
        anomaly_c, gat_probs_c = calc_custom.derive_scores(norm_custom)
        fused_c = (1 - alphas) * anomaly_c + alphas * gat_probs_c
        preds_c = (fused_c > 0.5).long()
        rewards_custom = calc_custom.compute(preds_c, labels, norm_custom, alphas)

        assert not torch.allclose(rewards_default, rewards_custom, atol=1e-3)

    def test_derive_scores_uses_vgae_weights(self):
        from graphids.core.models.fusion.fusion_reward import FusionRewardCalculator
        from graphids.core.models.fusion.fusion_features import fusion_state_dim

        state_dim = fusion_state_dim()
        torch.manual_seed(0)
        states = torch.rand(8, state_dim)

        calc_a = FusionRewardCalculator(vgae_weights=[1.0, 0.0, 0.0])
        calc_b = FusionRewardCalculator(vgae_weights=[0.0, 0.0, 1.0])

        scores_a, _ = calc_a.derive_scores(states)
        scores_b, _ = calc_b.derive_scores(states)

        assert not torch.allclose(scores_a, scores_b, atol=1e-4)


class TestFusionCheckpointRoundtrip:
    """Fusion checkpoint save/load format consistency."""

    def test_mlp_roundtrip(self, tmp_path):
        import pytorch_lightning as pl
        from graphids.core.models.fusion.fusion_baselines import MLPFusionModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim
        m1 = MLPFusionModule(state_dim=fusion_state_dim())
        m1.eval()
        trainer = pl.Trainer(enable_checkpointing=False, logger=False)
        trainer.strategy.connect(m1)
        ckpt_path = str(tmp_path / "mlp.ckpt")
        trainer.save_checkpoint(ckpt_path)
        m2 = MLPFusionModule.load_from_checkpoint(ckpt_path)
        m2.eval()
        x = torch.rand(8, fusion_state_dim())
        with torch.no_grad():
            torch.testing.assert_close(m1(x), m2(x))

    def test_weighted_avg_roundtrip(self, tmp_path):
        import pytorch_lightning as pl
        from graphids.core.models.fusion.fusion_baselines import WeightedAvgModule
        m1 = WeightedAvgModule()
        from graphids.core.models.fusion.fusion_features import fusion_state_dim
        m1.weight.data.fill_(0.7)
        m1.eval()
        trainer = pl.Trainer(enable_checkpointing=False, logger=False)
        trainer.strategy.connect(m1)
        ckpt_path = str(tmp_path / "wavg.ckpt")
        trainer.save_checkpoint(ckpt_path)
        m2 = WeightedAvgModule.load_from_checkpoint(ckpt_path)
        m2.eval()
        x = torch.rand(8, fusion_state_dim())
        with torch.no_grad():
            torch.testing.assert_close(m1(x), m2(x))

    def test_dqn_roundtrip(self, tmp_path):
        from graphids.core.models.fusion.dqn import DQNFusionModule
        from graphids.core.models.fusion.fusion_features import fusion_state_dim
        sd = fusion_state_dim()
        a1 = DQNFusionModule(
            alpha_steps=11, state_dim=sd,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        ckpt = {
            "q_network": a1.q_network.state_dict(),
            "target_network": a1.target_network.state_dict(),
            "epsilon": a1.epsilon,
        }
        torch.save(ckpt, tmp_path / "dqn.pt")
        a2 = DQNFusionModule(
            alpha_steps=11, state_dim=sd,
            reward_kwargs=dict(vgae_weights=[0.4, 0.35, 0.25]),
        )
        ckpt_loaded = torch.load(tmp_path / "dqn.pt", weights_only=True)
        a2.q_network.load_state_dict(ckpt_loaded["q_network"])
        a2.target_network.load_state_dict(ckpt_loaded["target_network"])
        a2.epsilon = ckpt_loaded["epsilon"]
        a1.q_network.eval()
        a2.q_network.eval()
        x = torch.rand(8, sd)
        with torch.no_grad():
            q1 = a1.q_network(x)
            q2 = a2.q_network(x)
        torch.testing.assert_close(q1, q2)
        assert a1.epsilon == a2.epsilon
