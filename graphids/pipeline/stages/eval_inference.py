"""Inference utilities for evaluation stage.

Metrics: handled by test_step + torchmetrics on VGAEModule/GATModule.
Artifacts: separate capture passes (embeddings, attention, components)
because they require different forward signatures or per-sample loops.
"""

from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import structlog
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGDataLoader

from dataclasses import dataclass

from .data_loading import graph_label


@dataclass(frozen=True)
class GATResult:
    preds: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    attack_types: np.ndarray
    embeddings: np.ndarray | None = None
    attention: list[dict] | None = None


@dataclass(frozen=True)
class VGAEResult:
    errors: np.ndarray
    labels: np.ndarray
    attack_types: np.ndarray
    embeddings: np.ndarray | None = None
    components: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class FusionResult:
    preds: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    q_values: np.ndarray

log = structlog.get_logger()

ATTENTION_SAMPLE_LIMIT = 50


# ---------------------------------------------------------------------------
# Test runner (uses test_step on training modules)
# ---------------------------------------------------------------------------

def _make_test_trainer() -> pl.Trainer:
    return pl.Trainer(
        accelerator="auto", devices="auto",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )


def extract_metrics(module) -> dict:
    """Extract computed test_metrics from a module into a standard dict."""
    r = module.test_metrics.compute()
    core = {k: v.item() for k, v in r.items()}
    core["balanced_accuracy"] = (core.get("recall", 0) + core.get("specificity", 0)) / 2
    return {"core": core, "additional": {}}


def test_model(module, data, batch_size: int = 128) -> dict:
    """Run trainer.test() on a module and return extracted metrics."""
    trainer = _make_test_trainer()
    loader = PyGDataLoader(data, batch_size=batch_size, shuffle=False)
    trainer.test(module, dataloaders=loader, verbose=False)
    return extract_metrics(module)


# ---------------------------------------------------------------------------
# VGAE threshold search
# ---------------------------------------------------------------------------

def find_vgae_threshold(module, data) -> tuple[float, float]:
    """Find optimal anomaly threshold via Youden's J on validation data.

    Runs test without a threshold to accumulate errors, then computes optimal.
    Returns (threshold, youden_j).
    """
    from torchmetrics.functional.classification import binary_roc

    module.test_threshold = None  # accumulate errors only
    module._test_errors.clear()
    module._test_labels.clear()

    trainer = _make_test_trainer()
    loader = PyGDataLoader(data, batch_size=128, shuffle=False)
    trainer.test(module, dataloaders=loader, verbose=False)

    errors, labels = module.get_test_errors()
    fpr_v, tpr_v, thresholds_v = binary_roc(
        torch.as_tensor(errors, dtype=torch.float),
        torch.as_tensor(labels, dtype=torch.long),
    )
    j_scores = tpr_v - fpr_v
    best_idx = torch.argmax(j_scores).item()
    thresh = float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(np.median(errors))
    return thresh, float(j_scores[best_idx])


# ---------------------------------------------------------------------------
# Artifact capture (separate passes — not compatible with test_step)
# ---------------------------------------------------------------------------

def capture_gat_artifacts(gat, data, device, embeddings: bool = True, attention: bool = True) -> GATResult:
    """Capture GAT embeddings and attention weights for paper artifacts."""
    preds_all, scores_all, labels_all, types_all, embs_all = [], [], [], [], []
    with torch.no_grad():
        for g in PyGDataLoader(data, batch_size=128, shuffle=False):
            g = g.to(device)
            logits, emb = gat(g, return_embedding=True)
            preds_all.append(logits.argmax(1).cpu())
            scores_all.append(F.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(g.y.cpu())
            at = g.attack_type.cpu() if hasattr(g, "attack_type") and g.attack_type is not None else torch.full((g.num_graphs,), -1)
            types_all.append(at)
            if embeddings:
                embs_all.append(emb.cpu())

    attn_data = None
    if attention:
        attn_data = []
        for idx in range(min(len(data), ATTENTION_SAMPLE_LIMIT)):
            g = data[idx].clone().to(device)
            with torch.no_grad():
                _, att_weights = gat(g, return_attention_weights=True)
            attn_data.append({
                "graph_idx": idx, "label": graph_label(g),
                "edge_index": g.edge_index.cpu().numpy(),
                "node_features": g.x[:, 0].cpu().numpy(),
                "attention_weights": [a.numpy() for a in att_weights],
            })

    return GATResult(
        preds=torch.cat(preds_all).numpy(), labels=torch.cat(labels_all).numpy(),
        scores=torch.cat(scores_all).numpy(), attack_types=torch.cat(types_all).numpy(),
        embeddings=torch.cat(embs_all).numpy() if embs_all else None,
        attention=attn_data,
    )


def capture_vgae_artifacts(vgae, data, device, embeddings: bool = True, components: bool = True) -> VGAEResult:
    """Capture VGAE embeddings and component-level loss decomposition."""
    from graphids.core.preprocessing import get_batch_index, graph_attack_type

    errors, labels, attack_types = [], [], []
    embs = [] if embeddings else None
    comps: dict[str, list] = {"recon": [], "canid": [], "nbr": [], "kl": []} if components else {}

    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)
            batch_idx = get_batch_index(g, device)
            edge_attr = getattr(g, "edge_attr", None)
            cont, canid_logits, nbr_logits, z, kl_loss = vgae(
                g.x, g.edge_index, batch_idx, edge_attr=edge_attr,
            )
            err = F.mse_loss(cont, g.x[:, 1:]).item()
            errors.append(err)
            labels.append(graph_label(g))
            attack_types.append(graph_attack_type(g))
            if embeddings and z is not None:
                embs.append(z.mean(dim=0).cpu().numpy())
            if components:
                comps["recon"].append(err)
                comps["canid"].append(F.cross_entropy(canid_logits, g.x[:, 0].long()).item())
                nbr_targets = vgae.create_neighborhood_targets(g.x, g.edge_index, batch_idx)
                comps["nbr"].append(F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets).item())
                comps["kl"].append(kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss))

    return VGAEResult(
        errors=np.array(errors), labels=np.array(labels), attack_types=np.array(attack_types),
        embeddings=np.array(embs) if embs else None,
        components={k: np.array(v) for k, v in comps.items()} if components else None,
    )


# ---------------------------------------------------------------------------
# Fusion inference (DQN agent, not a Lightning module)
# ---------------------------------------------------------------------------

def run_fusion_inference(agent, cache) -> FusionResult:
    """Run fusion inference (works for both DQN and bandit agents)."""
    states = cache["states"]
    labels_t = cache["labels"]
    actions, alphas, norm_states = agent.select_action_batch(states, training=False)
    anomaly_scores, gat_probs = agent.reward_calc.derive_scores(norm_states)
    fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
    preds = (fused_scores > 0.5).long()

    # DQN has q_network; bandit has theta (per-arm weights)
    with torch.no_grad():
        if hasattr(agent, "q_network"):
            q_values = agent.q_network(norm_states.to(agent.device)).cpu()
        else:
            z = agent.backbone(norm_states.to(agent.device))
            q_values = torch.einsum("kd,nd->nk", agent.theta, z).cpu()

    return FusionResult(
        preds=preds.numpy(), labels=labels_t.numpy(),
        scores=fused_scores.numpy(), q_values=q_values.numpy(),
    )
