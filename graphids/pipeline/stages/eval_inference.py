"""Inference functions for evaluation stage.

Common batched paths use Lightning trainer.predict() via predict_step on
GATModule/VGAEModule. Special capture modes (attention weights, VGAE
component decomposition) use manual per-sample loops because they require
different forward signatures incompatible with predict_step.

Design note — GATConv return type workaround:
  GATConv(return_attention_weights=True) changes the conv output type from
  Tensor to (Tensor, (Tensor, Tensor)). Lightning predict_step requires a
  consistent return type per batch. Solution: predict_step always uses
  return_embedding=True (consistent Tensor,Tensor return). Attention capture
  stays in a separate manual loop (_capture_attention).
"""

from __future__ import annotations

import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

from .eval_types import FusionResult, GATResult, VGAEResult
from .utils import graph_label

log = logging.getLogger(__name__)

ATTENTION_SAMPLE_LIMIT = 50  # Max graphs to capture attention for (export size)


# ---------------------------------------------------------------------------
# Lightweight prediction wrappers (wrap raw nn.Module for trainer.predict)
# ---------------------------------------------------------------------------


class _GATPredictor(pl.LightningModule):
    """Wraps a raw GAT model for batched prediction via trainer.predict()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, _batch_idx):
        logits, emb = self.model(batch, return_embedding=True)
        probs = F.softmax(logits, dim=1)
        return {
            "preds": logits.argmax(1).cpu(),
            "scores": probs[:, 1].cpu(),
            "labels": batch.y.cpu(),
            "attack_types": (
                batch.attack_type.cpu()
                if hasattr(batch, "attack_type") and batch.attack_type is not None
                else torch.full((batch.num_graphs,), -1)
            ),
            "embeddings": emb.cpu(),
        }


class _VGAEPredictor(pl.LightningModule):
    """Wraps a raw VGAE model for batched prediction via trainer.predict()."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, _batch_idx):
        from torch_geometric.nn import global_mean_pool
        from torch_geometric.utils import scatter

        edge_attr = getattr(batch, "edge_attr", None)
        cont, _, _, z, _ = self.model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)

        per_node_se = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
        graph_errors = scatter(per_node_se, batch.batch, dim=0, reduce="mean")

        result = {
            "errors": graph_errors.cpu(),
            "labels": batch.y.cpu(),
            "attack_types": (
                batch.attack_type.cpu()
                if hasattr(batch, "attack_type") and batch.attack_type is not None
                else torch.full((batch.num_graphs,), -1)
            ),
        }
        if z is not None:
            result["embeddings"] = global_mean_pool(z, batch.batch).cpu()
        return result


def _make_predict_trainer() -> pl.Trainer:
    """Create a minimal Trainer for prediction only (no logging, no checkpoints)."""
    return pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )


def _collate_predictions(batch_results: list[dict], keys: list[str]) -> dict[str, np.ndarray]:
    """Collate list of per-batch dicts into concatenated numpy arrays."""
    collated = {}
    for key in keys:
        tensors = [b[key] for b in batch_results if key in b]
        if tensors:
            collated[key] = torch.cat(tensors, dim=0).numpy()
    return collated


# ---------------------------------------------------------------------------
# GAT inference
# ---------------------------------------------------------------------------


def run_gat_inference(
    gat,
    data,
    device,
    capture_embeddings: bool = False,
    capture_attention: bool = False,
) -> GATResult:
    """Run GAT inference. Batched via trainer.predict(); attention capture is per-sample."""
    # Batched common path via Lightning predict
    predictor = _GATPredictor(gat)
    trainer = _make_predict_trainer()
    loader = PyGDataLoader(data, batch_size=128, shuffle=False)
    batch_results = trainer.predict(predictor, loader)

    collated = _collate_predictions(batch_results, ["preds", "scores", "labels", "attack_types", "embeddings"])

    # Attention capture (separate per-sample pass, small subset only)
    attn_data = None
    if capture_attention:
        attn_data = _capture_attention(gat, data, device)

    return GATResult(
        preds=collated["preds"],
        labels=collated["labels"],
        scores=collated["scores"],
        attack_types=collated["attack_types"],
        embeddings=collated["embeddings"] if capture_embeddings else None,
        attention=attn_data,
    )


def _capture_attention(gat, data, device) -> list[dict]:
    """Per-sample attention weight extraction (small subset).

    Uses return_attention_weights=True which changes GATConv's return type —
    this is why it can't be in predict_step.
    """
    attn_data = []
    for idx in range(min(len(data), ATTENTION_SAMPLE_LIMIT)):
        g = data[idx].clone().to(device)
        with torch.no_grad():
            _, att_weights = gat(g, return_attention_weights=True)
        attn_data.append(
            {
                "graph_idx": idx,
                "label": graph_label(g),
                "edge_index": g.edge_index.cpu().numpy(),
                "node_features": g.x[:, 0].cpu().numpy(),
                "attention_weights": [a.numpy() for a in att_weights],
            }
        )
    return attn_data


# ---------------------------------------------------------------------------
# VGAE inference
# ---------------------------------------------------------------------------


def run_vgae_inference(
    vgae,
    data,
    device,
    capture_embeddings: bool = False,
    capture_components: bool = False,
) -> VGAEResult:
    """Run VGAE reconstruction-error inference.

    Batched via trainer.predict() for the common path.
    Falls back to per-sample when capture_components=True (needs per-graph
    neighborhood targets and KL decomposition).
    """
    if capture_components:
        return _run_vgae_inference_per_sample(vgae, data, device, capture_embeddings)

    # Batched common path via Lightning predict
    predictor = _VGAEPredictor(vgae)
    trainer = _make_predict_trainer()
    loader = PyGDataLoader(data, batch_size=128, shuffle=False)
    batch_results = trainer.predict(predictor, loader)

    collated = _collate_predictions(batch_results, ["errors", "labels", "attack_types", "embeddings"])

    return VGAEResult(
        errors=collated["errors"],
        labels=collated["labels"],
        attack_types=collated["attack_types"],
        embeddings=collated["embeddings"] if capture_embeddings else None,
        components=None,
    )


def _run_vgae_inference_per_sample(vgae, data, device, capture_embeddings: bool) -> VGAEResult:
    """Per-sample VGAE inference with component-level loss decomposition.

    Manual loop because KL decomposition and neighborhood target creation
    require per-graph batch indices — not compatible with batched predict_step.
    """
    from graphids.core.preprocessing import get_batch_index, graph_attack_type

    errors, labels, attack_types = [], [], []
    embeddings = [] if capture_embeddings else None
    components: dict[str, list] = {"recon": [], "canid": [], "nbr": [], "kl": []}

    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)
            batch_idx = get_batch_index(g, device)
            edge_attr = getattr(g, "edge_attr", None)
            cont, canid_logits, nbr_logits, z, kl_loss = vgae(
                g.x, g.edge_index, batch_idx, edge_attr=edge_attr
            )
            err = F.mse_loss(cont, g.x[:, 1:]).item()
            errors.append(err)
            labels.append(graph_label(g))
            attack_types.append(graph_attack_type(g))
            if capture_embeddings and z is not None:
                embeddings.append(z.mean(dim=0).cpu().numpy())

            components["recon"].append(err)
            components["canid"].append(F.cross_entropy(canid_logits, g.x[:, 0].long()).item())
            nbr_targets = vgae.create_neighborhood_targets(g.x, g.edge_index, batch_idx)
            components["nbr"].append(
                F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets).item()
            )
            components["kl"].append(kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss))

    return VGAEResult(
        errors=np.array(errors),
        labels=np.array(labels),
        attack_types=np.array(attack_types),
        embeddings=np.array(embeddings) if capture_embeddings and embeddings else None,
        components={k: np.array(v) for k, v in components.items()},
    )


# ---------------------------------------------------------------------------
# Fusion inference (not a Lightning module — stays manual)
# ---------------------------------------------------------------------------


def run_fusion_inference(agent, cache) -> FusionResult:
    """Run DQN fusion inference (vectorized)."""
    states = cache["states"]  # [N, D] tensor
    labels_t = cache["labels"]  # [N] tensor

    actions, alphas, norm_states = agent.select_action_batch(states, training=False)
    anomaly_scores, gat_probs = agent._derive_scores_batch(norm_states)
    fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
    preds = (fused_scores > 0.5).long()

    with torch.no_grad():
        q_values = agent.q_network(norm_states.to(agent.device)).cpu()

    return FusionResult(
        preds=preds.numpy(),
        labels=labels_t.numpy(),
        scores=fused_scores.numpy(),
        q_values=q_values.numpy(),
    )
