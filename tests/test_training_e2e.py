"""End-to-end tests for training stages.

Tests the full train_autoencoder -> train_curriculum -> train_fusion pipeline
with synthetic data on CPU. Verifies that checkpoints and configs are saved
correctly and that downstream stages can load upstream outputs.

Run:  python -m pytest tests/test_training_e2e.py -v -m "not slow"
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from tests.conftest import E2E_OVERRIDES, IN_CHANNELS, NUM_IDS, _make_dataset


def _patch_load_data(data):
    """Return a monkeypatch-ready load_data that returns synthetic data."""

    def _fake_load_data(cfg):
        train = data[:35]
        val = data[35:]
        return train, val, NUM_IDS, IN_CHANNELS

    return _fake_load_data


@pytest.fixture()
def synth_data():
    return _make_dataset(50)


@pytest.fixture()
def exp_root(tmp_path):
    return str(tmp_path / "experimentruns")


def _apply_load_data_patches(stack, data):
    """Patch load_data at canonical source + all re-import sites."""
    fake = _patch_load_data(data)
    stack.enter_context(patch("graphids.pipeline.stages.data_loading.load_data", fake))
    stack.enter_context(patch("graphids.pipeline.stages.training.load_data", fake))


@pytest.mark.slurm
class TestAutoencoderE2E:
    """train_autoencoder produces checkpoint + config that load correctly."""

    def test_autoencoder_e2e(self, synth_data, exp_root):
        from graphids.config import config_path, resolve
        from graphids.pipeline.stages.training import train_autoencoder

        cfg = resolve(
            "vgae",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            result = train_autoencoder(cfg)

        ckpt = Path(result["checkpoint"])
        assert ckpt.exists(), "Checkpoint not saved"
        assert config_path(cfg, "autoencoder").exists(), "Config not saved"

        # Verify checkpoint loads back
        from graphids.config import PipelineConfig

        loaded_cfg = PipelineConfig.load(config_path(cfg, "autoencoder"))
        from graphids.core.models.vgae import GraphAutoencoderNeighborhood

        model = GraphAutoencoderNeighborhood.from_config(loaded_cfg, NUM_IDS, IN_CHANNELS)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))


@pytest.mark.slurm
class TestCurriculumE2E:
    """train_curriculum loads VGAE, trains GAT, saves checkpoint."""

    def test_curriculum_e2e(self, synth_data, exp_root):
        from graphids.config import config_path, resolve
        from graphids.pipeline.stages.training import (
            train_autoencoder,
            train_curriculum,
        )

        vgae_cfg = resolve(
            "vgae",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )
        gat_cfg = resolve(
            "gat",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(vgae_cfg)
            result = train_curriculum(gat_cfg)

        ckpt = Path(result["checkpoint"])
        assert ckpt.exists(), "GAT checkpoint not saved"
        assert config_path(gat_cfg, "curriculum").exists(), "GAT config not saved"

        from graphids.config import PipelineConfig

        loaded_cfg = PipelineConfig.load(config_path(gat_cfg, "curriculum"))
        from graphids.core.models.gat import GATWithJK

        model = GATWithJK.from_config(loaded_cfg, NUM_IDS, IN_CHANNELS)
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))


@pytest.mark.slurm
class TestFusionE2E:
    """train_fusion loads VGAE+GAT, trains DQN, saves checkpoint."""

    def test_fusion_e2e(self, synth_data, exp_root):
        from graphids.config import config_path, resolve
        from graphids.pipeline.stages.fusion import train_fusion
        from graphids.pipeline.stages.training import (
            train_autoencoder,
            train_curriculum,
        )

        vgae_cfg = resolve(
            "vgae",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )
        gat_cfg = resolve(
            "gat",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )
        dqn_cfg = resolve(
            "dqn",
            "large",
            dataset="test_ds",
            lake_root=exp_root,
            fusion=dict(
                episodes=5,
                episode_sample_size=20,
                max_samples=50,
                max_val_samples=15,
                gpu_training_steps=2,
            ),
            **E2E_OVERRIDES,
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(vgae_cfg)
            train_curriculum(gat_cfg)
            result = train_fusion(dqn_cfg)

        ckpt = Path(result["checkpoint"])
        assert ckpt.exists(), "DQN checkpoint not saved"
        assert config_path(dqn_cfg, "fusion").exists(), "DQN config not saved"

        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        assert "q_network" in sd


@pytest.mark.slow
@pytest.mark.slurm
class TestFullPipelineE2E:
    """Full 3-stage pipeline + evaluation in sequence."""

    def test_full_pipeline(self, synth_data, exp_root):
        from graphids.config import resolve, stage_dir
        from graphids.pipeline.stages.evaluation import evaluate
        from graphids.pipeline.stages.fusion import train_fusion
        from graphids.pipeline.stages.training import (
            train_autoencoder,
            train_curriculum,
        )

        common = dict(
            dataset="test_ds",
            lake_root=exp_root,
            **E2E_OVERRIDES,
        )
        fusion_overrides = dict(
            fusion=dict(
                episodes=3,
                episode_sample_size=20,
                max_samples=50,
                max_val_samples=15,
                gpu_training_steps=2,
            ),
        )

        with ExitStack() as stack:
            _apply_load_data_patches(stack, synth_data)
            train_autoencoder(resolve("vgae", "large", **common))
            train_curriculum(resolve("gat", "large", **common))
            train_fusion(resolve("dqn", "large", **fusion_overrides, **common))

            eval_cfg = resolve(
                "vgae",
                "large",
                dataset="test_ds",
                lake_root=exp_root,
                **E2E_OVERRIDES,
            )

            stack.enter_context(
                patch("graphids.pipeline.stages.evaluation._load_test_data", return_value={})
            )
            result = evaluate(eval_cfg)

        metrics = result["metrics"]
        assert "gat" in metrics, "GAT metrics missing"
        assert "vgae" in metrics, "VGAE metrics missing"
        assert "fusion" in metrics, "Fusion metrics missing"
        assert metrics["gat"]["core"]["accuracy"] >= 0.0
        assert (stage_dir(eval_cfg, "evaluation") / "_manifest.json").exists()
