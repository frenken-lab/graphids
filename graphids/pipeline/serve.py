"""FastAPI inference server for KD-GAT fusion predictions.

Loads trained VGAE + GAT + DQN models at startup and serves predictions
via REST API. Designed for deployment on OSC OnDemand or forwarded ports.

Usage:
    uvicorn pipeline.serve:app --host 0.0.0.0 --port 8000
    # Or with reload for development:
    uvicorn pipeline.serve:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    """CAN frame window for classification."""

    node_features: list[list[float]] = Field(
        ...,
        max_length=1000,
        description="Node feature matrix [num_nodes, num_features]",
    )
    edge_index: list[list[int]] = Field(
        ...,
        max_length=10000,
        description="Edge index [2, num_edges] as list of [src, dst] pairs",
    )
    dataset: str = Field(default="hcrl_sa", description="Dataset the models were trained on")
    scale: str = Field(
        default="large",
        pattern=r"^(large|small)$",
        description="Model scale: large or small",
    )


class PredictResponse(BaseModel):
    """Classification result with confidence and fusion alpha."""

    prediction: int = Field(description="0=normal, 1=attack")
    label: str = Field(description="'normal' or 'attack'")
    confidence: float = Field(description="Softmax probability of predicted class")
    alpha: float = Field(description="DQN fusion weight (GAT vs VGAE blend)")
    gat_prob: float = Field(description="GAT attack probability")
    vgae_error: float = Field(description="VGAE reconstruction error")


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
    device: str


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_models: dict = {}
_device: torch.device = torch.device("cpu")


def _load_models(dataset: str, scale: str) -> dict:
    """Load VGAE + GAT + DQN using ArtifactResolver."""
    from graphids.config import get_resolver
    from graphids.config.resolver import resolve
    from graphids.pipeline.stages.utils import load_data, load_model

    cache_key = f"{dataset}_{scale}"
    if cache_key in _models:
        return _models[cache_key]

    cfg = resolve("vgae", scale, dataset=dataset)
    _, _, num_ids, in_ch = load_data(cfg)
    resolver = get_resolver()

    models = {}

    # VGAE
    if resolver.exists(cfg, "autoencoder", "best_model.pt", model_type="vgae"):
        models["vgae"] = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, _device)

    # GAT
    gat_cfg = resolve("gat", scale, dataset=dataset)
    if resolver.exists(gat_cfg, "curriculum", "best_model.pt", model_type="gat"):
        models["gat"] = load_model(gat_cfg, "gat", "curriculum", num_ids, in_ch, _device)

    # DQN
    dqn_cfg = resolve("dqn", scale, dataset=dataset)
    if resolver.exists(dqn_cfg, "fusion", "best_model.pt", model_type="dqn"):
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.pipeline.stages.trainer_factory import load_frozen_cfg

        fusion_cfg = load_frozen_cfg(dqn_cfg, "fusion")
        fusion_ckpt = resolver.get(dqn_cfg, "fusion", "best_model.pt", model_type="dqn")
        agent = EnhancedDQNFusionAgent.from_config(fusion_cfg, device=str(_device), inference=True)
        agent.load_checkpoint(fusion_ckpt)
        models["dqn"] = agent

    _models[cache_key] = models
    log.info("Loaded models for %s/%s: %s", dataset, scale, list(models.keys()))
    return models


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default models at startup."""
    global _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Inference device: %s", _device)
    yield
    _models.clear()


app = FastAPI(
    title="KD-GAT Inference Server",
    description="CAN bus intrusion detection via VGAE+GAT+DQN fusion",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = []
    for key, models in _models.items():
        for model_name in models:
            loaded.append(f"{key}/{model_name}")
    return HealthResponse(
        status="ok",
        models_loaded=loaded,
        device=str(_device),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    from torch_geometric.data import Data

    try:
        models = _load_models(req.dataset, req.scale)
    except Exception as exc:
        log.exception("Model loading failed for %s/%s", req.dataset, req.scale)
        raise HTTPException(
            status_code=503,
            detail=f"Model loading failed for {req.dataset}/{req.scale}",
        ) from exc

    if "gat" not in models or "vgae" not in models:
        raise HTTPException(
            status_code=503,
            detail=f"Required models not available for {req.dataset}/{req.scale}",
        )

    # Build PyG Data object
    x = torch.tensor(req.node_features, dtype=torch.float32)
    edge_index = torch.tensor(req.edge_index, dtype=torch.long)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.t().contiguous()
    data = Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))
    data = data.to(_device)

    with torch.no_grad():
        batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=_device)
        # Build 15-D state using registry extractors (VGAE 8-D + GAT 7-D)
        from graphids.core.models.registry import extractors as get_extractors

        features = []
        for name, extractor in get_extractors():
            feat = extractor.extract(models[name], data, batch_idx, _device)
            features.append(feat)
        state = torch.cat(features).numpy()

    # DQN fusion
    if "dqn" not in models:
        raise HTTPException(status_code=503, detail="DQN model not available for fusion")
    agent = models["dqn"]
    alpha, _, _ = agent.select_action(state, training=False)
    anomaly_score, gat_prob = agent._derive_scores(state)

    fused_score = (1 - alpha) * anomaly_score + alpha * gat_prob
    prediction = 1 if fused_score > 0.5 else 0

    return PredictResponse(
        prediction=prediction,
        label="attack" if prediction == 1 else "normal",
        confidence=max(fused_score, 1.0 - fused_score),
        alpha=float(alpha),
        gat_prob=gat_prob,
        vgae_error=anomaly_score,
    )
