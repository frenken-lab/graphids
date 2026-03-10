"""Mock-based tests for pipeline.serve — no GPU or SLURM required."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient

from graphids.pipeline.serve import _models, app


@pytest.fixture()
def client():
    """TestClient that clears model cache before and after."""
    _models.clear()
    with TestClient(app) as c:
        yield c
    _models.clear()


def _make_mock_models() -> dict:
    """Return dict with mock VGAE, GAT, and DQN models."""
    # Mock VGAE extractor output: 8-D feature vector
    vgae = MagicMock()

    # Mock GAT extractor output: 7-D feature vector
    gat = MagicMock()

    # Mock DQN agent
    dqn = MagicMock()
    dqn.state_dim = 15
    dqn.select_action.return_value = (0.7, 5, None)
    dqn._derive_scores.return_value = (0.3, 0.8)

    return {"vgae": vgae, "gat": gat, "dqn": dqn}


def _sample_request() -> dict:
    """Minimal valid predict request."""
    return {
        "node_features": [[1.0, 0.5, 0.3] for _ in range(5)],
        "edge_index": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
        "dataset": "hcrl_sa",
        "scale": "large",
    }


# ---- Tests ----


def test_health_empty(client):
    """GET /health → 200 with empty models_loaded."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["models_loaded"] == []


def test_health_schema(client):
    """Health response has required fields."""
    resp = client.get("/health")
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "device" in data


def test_predict_no_models_503(client):
    """POST /predict → 503 when no checkpoints found."""
    with patch("graphids.pipeline.serve._load_models", return_value={}):
        resp = client.post("/predict", json=_sample_request())
    assert resp.status_code == 503


def test_predict_mocked(client):
    """Mocked prediction returns all response fields."""
    models = _make_mock_models()

    # Mock the extractors to return proper tensors
    mock_extractors = [
        ("vgae", MagicMock()),
        ("gat", MagicMock()),
    ]
    mock_extractors[0][1].extract.return_value = torch.tensor(
        [0.3, 0.1, 0.2, 0.0, 0.5, 1.0, -1.0, 0.77]
    )
    mock_extractors[1][1].extract.return_value = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.5, -0.1, 0.9])

    with (
        patch("graphids.pipeline.serve._load_models", return_value=models),
        patch("graphids.core.models.registry.extractors", return_value=mock_extractors),
    ):
        resp = client.post("/predict", json=_sample_request())

    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data
    assert data["prediction"] in (0, 1)
    assert "label" in data
    assert data["label"] in ("normal", "attack")
    assert "confidence" in data
    assert 0.5 <= data["confidence"] <= 1.0
    assert "alpha" in data
    assert "gat_prob" in data
    assert "vgae_error" in data


def test_predict_without_dqn(client):
    """Without DQN, prediction returns 503 (DQN required for fusion)."""
    models = {"vgae": MagicMock(), "gat": MagicMock()}

    mock_extractors = [
        ("vgae", MagicMock()),
        ("gat", MagicMock()),
    ]
    mock_extractors[0][1].extract.return_value = torch.tensor(
        [0.3, 0.1, 0.2, 0.0, 0.5, 1.0, -1.0, 0.77]
    )
    mock_extractors[1][1].extract.return_value = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.5, -0.1, 0.9])

    with (
        patch("graphids.pipeline.serve._load_models", return_value=models),
        patch("graphids.core.models.registry.extractors", return_value=mock_extractors),
    ):
        resp = client.post("/predict", json=_sample_request())

    assert resp.status_code == 503
    assert "DQN" in resp.json()["detail"]


def test_predict_edge_index_transpose(client):
    """[N,2] edge_index input gets transposed to [2,N]."""
    models = _make_mock_models()

    mock_extractors = [
        ("vgae", MagicMock()),
        ("gat", MagicMock()),
    ]
    mock_extractors[0][1].extract.return_value = torch.tensor(
        [0.3, 0.1, 0.2, 0.0, 0.5, 1.0, -1.0, 0.77]
    )
    mock_extractors[1][1].extract.return_value = torch.tensor([0.2, 0.8, 0.1, 0.3, 0.5, -0.1, 0.9])

    # edge_index as [N, 2] (common user format) — should be transposed
    req = _sample_request()
    req["edge_index"] = [[0, 1], [1, 2], [2, 3]]  # shape [3, 2]

    with (
        patch("graphids.pipeline.serve._load_models", return_value=models),
        patch("graphids.core.models.registry.extractors", return_value=mock_extractors),
    ):
        resp = client.post("/predict", json=req)

    assert resp.status_code == 200


def test_request_validation(client):
    """Malformed body → 422."""
    resp = client.post("/predict", json={"bad": "data"})
    assert resp.status_code == 422
