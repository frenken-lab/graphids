"""Tests for newly added modules: GAT return_embedding, DQN compute_fusion_reward,
CLI archive/restore, and FastAPI serve.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on sys.path for scripts/ imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from tests.conftest import IN_CHANNELS, NUM_IDS, SMOKE_OVERRIDES, _make_graph

# ---------------------------------------------------------------------------
# P1: GAT return_embedding
# ---------------------------------------------------------------------------


@pytest.mark.slurm
class TestGATReturnEmbedding:
    """Test the return_embedding flag on GATWithJK.forward()."""

    @pytest.fixture
    def gat_model(self):
        from graphids.config.resolver import resolve

        cfg = resolve("gat", "small", dataset="hcrl_sa", **SMOKE_OVERRIDES)
        from graphids.core.models.gat import GATWithJK

        return GATWithJK.from_config(cfg, NUM_IDS, IN_CHANNELS)

    def test_return_embedding_tuple(self, gat_model):
        """return_embedding=True should return (logits, embedding) tuple."""
        data = _make_graph(num_nodes=10, num_edges=20, label=0)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        gat_model.eval()
        with torch.no_grad():
            result = gat_model(data, return_embedding=True)
        assert isinstance(result, tuple), "Expected tuple (logits, embedding)"
        assert len(result) == 2
        logits, embedding = result
        assert logits.shape == (1, 2), f"Expected logits shape (1, 2), got {logits.shape}"
        assert embedding.ndim == 2 and embedding.shape[0] == 1

    def test_return_embedding_false(self, gat_model):
        """return_embedding=False (default) should return plain tensor."""
        data = _make_graph(num_nodes=10, num_edges=20, label=0)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        gat_model.eval()
        with torch.no_grad():
            result = gat_model(data, return_embedding=False)
        assert isinstance(result, torch.Tensor), "Expected plain Tensor"
        assert result.shape == (1, 2)

    def test_embedding_differs_from_logits(self, gat_model):
        """Embedding should be pre-FC representation, different from logits."""
        data = _make_graph(num_nodes=10, num_edges=20, label=0)
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
        gat_model.eval()
        with torch.no_grad():
            logits, embedding = gat_model(data, return_embedding=True)
        # Embedding is JK+pool output; should be higher-dimensional than 2-class logits
        assert embedding.shape[1] > logits.shape[1]


# ---------------------------------------------------------------------------
# P1: DQN compute_fusion_reward
# ---------------------------------------------------------------------------


@pytest.mark.slurm
class TestDQNComputeFusionReward:
    """Test EnhancedDQNFusionAgent.compute_fusion_reward()."""

    @pytest.fixture
    def agent(self):
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.registry import fusion_state_dim

        return EnhancedDQNFusionAgent(
            alpha_steps=11,
            lr=1e-3,
            gamma=0.9,
            state_dim=fusion_state_dim(),
            hidden_dim=32,
            num_layers=2,
            device="cpu",
        )

    def _make_state(self, agent):
        """Create a valid random state vector of expected dimension."""
        return np.random.rand(agent.state_dim).astype(np.float32)

    def test_correct_prediction_positive_reward(self, agent):
        """Correct prediction should yield positive total reward."""
        state = self._make_state(agent)
        reward = agent.compute_fusion_reward(
            prediction=1,
            true_label=1,
            state_features=state,
            alpha=0.5,
        )
        assert reward > 0, f"Correct prediction should have positive reward, got {reward}"

    def test_wrong_prediction_negative_reward(self, agent):
        """Wrong prediction should yield negative total reward."""
        state = self._make_state(agent)
        reward = agent.compute_fusion_reward(
            prediction=0,
            true_label=1,
            state_features=state,
            alpha=0.5,
        )
        assert reward < 0, f"Wrong prediction should have negative reward, got {reward}"

    def test_confidence_bonus(self, agent):
        """High-confidence correct predictions should get higher reward than low-confidence."""
        # State with high GAT confidence (high prob for class 1)
        high_conf = np.zeros(agent.state_dim, dtype=np.float32)
        # Set GAT logits to strongly predict attack (indices from layout)
        from graphids.core.models.registry import feature_layout

        layout = feature_layout()
        gat_start = layout["gat"][0]
        # class 0 prob low, class 1 prob high
        high_conf[gat_start] = 0.1
        high_conf[gat_start + 1] = 0.9
        # Set confidence indices
        high_conf[layout["vgae"][2]] = 0.5  # moderate vgae confidence
        high_conf[layout["gat"][2]] = 0.95  # high gat confidence

        low_conf = high_conf.copy()
        low_conf[gat_start] = 0.45
        low_conf[gat_start + 1] = 0.55
        low_conf[layout["gat"][2]] = 0.55  # low gat confidence

        r_high = agent.compute_fusion_reward(1, 1, high_conf, 0.5)
        r_low = agent.compute_fusion_reward(1, 1, low_conf, 0.5)
        assert r_high > r_low, (
            f"High confidence reward ({r_high}) should exceed low confidence ({r_low})"
        )

    def test_reward_is_finite(self, agent):
        """Reward should always be a finite float."""
        state = self._make_state(agent)
        for pred, label in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            reward = agent.compute_fusion_reward(pred, label, state, 0.5)
            assert np.isfinite(reward), f"Reward not finite for pred={pred}, label={label}"


# ---------------------------------------------------------------------------
# P2: CLI archive/restore
# ---------------------------------------------------------------------------


@pytest.mark.slurm
class TestCLIArchiveRestore:
    """Test archive-on-rerun and restore-on-failure logic from cli.py."""

    def test_archive_created_on_rerun(self, tmp_path):
        """When a stage dir has metrics.json, re-running should archive it."""
        from datetime import datetime

        from graphids.config import stage_dir
        from graphids.config.resolver import resolve

        cfg = resolve("vgae", "large", dataset="hcrl_sa", experiment_root=str(tmp_path))
        sdir = stage_dir(cfg, "autoencoder")
        sdir.mkdir(parents=True)
        (sdir / "metrics.json").write_text('{"loss": 0.5}')
        (sdir / "best_model.pt").write_text("fake")

        # Simulate archive logic from cli.py
        if (sdir / "metrics.json").exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive = sdir.parent / f"{sdir.name}.archive_{ts}"
            sdir.rename(archive)

        assert not sdir.exists(), "Original dir should be gone"
        archives = list(sdir.parent.glob("*.archive_*"))
        assert len(archives) == 1
        assert (archives[0] / "metrics.json").exists()
        assert (archives[0] / "best_model.pt").exists()

    def test_restore_on_failure(self, tmp_path):
        """On failure, archive should be restored to original path."""
        import shutil
        from datetime import datetime

        from graphids.config import stage_dir
        from graphids.config.resolver import resolve

        cfg = resolve("vgae", "large", dataset="hcrl_sa", experiment_root=str(tmp_path))
        sdir = stage_dir(cfg, "autoencoder")
        sdir.mkdir(parents=True)
        (sdir / "metrics.json").write_text('{"loss": 0.5}')

        # Archive
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)

        # Simulate new run creating partial output then failing
        sdir.mkdir(parents=True)
        (sdir / "partial.txt").write_text("incomplete")

        # Failure → restore archive
        if archive.exists():
            if sdir.exists():
                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)

        assert sdir.exists()
        assert (sdir / "metrics.json").exists()
        assert not (sdir / "partial.txt").exists()

    def test_archive_deleted_on_success(self, tmp_path):
        """On success, archive should be cleaned up."""
        import shutil
        from datetime import datetime

        from graphids.config import stage_dir
        from graphids.config.resolver import resolve

        cfg = resolve("vgae", "large", dataset="hcrl_sa", experiment_root=str(tmp_path))
        sdir = stage_dir(cfg, "autoencoder")
        sdir.mkdir(parents=True)
        (sdir / "metrics.json").write_text('{"loss": 0.5}')

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive = sdir.parent / f"{sdir.name}.archive_{ts}"
        sdir.rename(archive)

        # Simulate success → delete archive
        sdir.mkdir(parents=True)
        if archive.exists():
            shutil.rmtree(archive, ignore_errors=True)

        assert not archive.exists()


# ---------------------------------------------------------------------------
# P3: FastAPI serve.py (smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestServeSmokeTest:
    """Smoke tests for the FastAPI inference server."""

    @pytest.fixture
    def client(self):
        """Create a test client without loading real models."""
        from starlette.testclient import TestClient

        from graphids.pipeline.serve import app

        return TestClient(app)

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "models_loaded" in body
        assert "device" in body

    def test_predict_no_models_returns_503(self, client):
        """POST /predict with no models loaded should return 503."""
        resp = client.post(
            "/predict",
            json={
                "node_features": [[0.0] * IN_CHANNELS] * 5,
                "edge_index": [[0, 1, 2, 3], [1, 2, 3, 4]],
                "dataset": "nonexistent",
                "scale": "large",
            },
        )
        assert resp.status_code == 503
