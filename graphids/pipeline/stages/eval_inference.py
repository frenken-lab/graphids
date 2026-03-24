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


def graph_label(g) -> int:
    """Extract scalar label from a PyG Data object (handles 0-D and 1-D y)."""
    return g.y.item() if g.y.dim() == 0 else int(g.y[0].item())


# ---------------------------------------------------------------------------
# Test runner (uses test_step on training modules)
# ---------------------------------------------------------------------------

def make_test_trainer() -> pl.Trainer:
    return pl.Trainer(
        accelerator="auto", devices="auto",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )


def test_model(module, data, batch_size: int = 256) -> dict:
    """Run trainer.test() on a module and return metrics from the test loop.

    Uses the return value of ``trainer.test()`` which contains all metrics
    logged via ``self.log_dict()`` in each module's ``on_test_epoch_end``.

    Args:
        data: Either a list of PyG Data objects (creates PyGDataLoader) or
              a pre-built DataLoader (used as-is, e.g. for fusion tensor batches).
    """
    trainer = make_test_trainer()
    if isinstance(data, list):
        loader = PyGDataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        loader = data  # pre-built DataLoader (fusion, temporal)
    results = trainer.test(module, dataloaders=loader, verbose=False)
    metrics = dict(results[0]) if results else {}
    metrics["balanced_accuracy"] = (metrics.get("recall", 0) + metrics.get("specificity", 0)) / 2
    return metrics


# ---------------------------------------------------------------------------
# VGAE threshold search
# ---------------------------------------------------------------------------

def find_vgae_threshold(module, data, batch_size: int = 256) -> tuple[float, float]:
    """Find optimal anomaly threshold via Youden's J on validation data.

    Runs test without a threshold to accumulate errors, then computes optimal.
    Returns (threshold, youden_j).
    """
    from torchmetrics.functional.classification import binary_roc

    module.test_threshold = None  # accumulate errors only
    module._test_errors.clear()
    module._test_labels.clear()

    trainer = make_test_trainer()
    loader = PyGDataLoader(data, batch_size=batch_size, shuffle=False)
    trainer.test(module, dataloaders=loader, verbose=False)

    errors, labels = module.get_test_errors()

    # Guard: no data processed
    if len(errors) == 0:
        log.warning("find_vgae_threshold_no_data")
        return 0.5, 0.0

    # Guard: single-class labels (ROC undefined)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        log.warning("find_vgae_threshold_single_class", unique_label=int(unique_labels[0]))
        return float(np.median(errors)), 0.0

    fpr_v, tpr_v, thresholds_v = binary_roc(
        torch.as_tensor(errors, dtype=torch.float),
        torch.as_tensor(labels, dtype=torch.long),
    )
    j_scores = tpr_v - fpr_v

    # Guard: degenerate ROC curve
    if len(j_scores) == 0 or len(thresholds_v) == 0:
        return float(np.median(errors)), 0.0

    best_idx = torch.argmax(j_scores).item()
    thresh = float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(np.median(errors))
    return thresh, float(j_scores[best_idx])


# ---------------------------------------------------------------------------
# Artifact capture (separate passes — not compatible with test_step)
# ---------------------------------------------------------------------------

def capture_gat_artifacts(gat, data, device, embeddings: bool = True, attention: bool = True, batch_size: int = 256, attention_limit: int = 50) -> GATResult:
    """Capture GAT embeddings and attention weights for paper artifacts."""
    preds_all, scores_all, labels_all, types_all, embs_all = [], [], [], [], []
    with torch.no_grad():
        for g in PyGDataLoader(data, batch_size=batch_size, shuffle=False):
            g = g.to(device, non_blocking=True)
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
        for idx in range(min(len(data), attention_limit)):
            g = data[idx].clone().to(device, non_blocking=True)
            with torch.no_grad():
                _, att_weights = gat(g, return_attention_weights=True)
            attn_data.append({
                "graph_idx": idx, "label": graph_label(g),
                "edge_index": g.edge_index.cpu().numpy(),
                "node_features": g.node_id.cpu().numpy(),
                "attention_weights": [a.numpy() for a in att_weights],
            })

    return GATResult(
        preds=torch.cat(preds_all).numpy(), labels=torch.cat(labels_all).numpy(),
        scores=torch.cat(scores_all).numpy(), attack_types=torch.cat(types_all).numpy(),
        embeddings=torch.cat(embs_all).numpy() if embs_all else None,
        attention=attn_data,
    )


def capture_vgae_artifacts(vgae, data, device, embeddings: bool = True, components: bool = True, batch_size: int = 256) -> VGAEResult:
    """Capture VGAE embeddings and component-level loss decomposition.

    Uses batched forward passes with scatter reduction for per-graph losses.
    """
    from graphids.core.preprocessing import graph_attack_type
    from torch_geometric.utils import scatter

    errors_all, labels_all, types_all = [], [], []
    embs_all = [] if embeddings else None
    comps: dict[str, list] = {"recon": [], "canid": [], "nbr": [], "kl": []} if components else {}

    with torch.no_grad():
        for batch in PyGDataLoader(data, batch_size=batch_size, shuffle=False):
            batch = batch.to(device, non_blocking=True)
            edge_attr = getattr(batch, "edge_attr", None)
            cont, canid_logits, nbr_logits, z, kl_loss, _ = vgae(
                batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr,
                node_id=batch.node_id,
            )
            # Per-graph MSE via scatter
            node_mse = (cont - batch.x).pow(2).mean(dim=1)
            graph_mse = scatter(node_mse, batch.batch, reduce="mean")
            errors_all.append(graph_mse.cpu())

            # Per-graph labels and attack types
            for g in batch.to_data_list():
                labels_all.append(graph_label(g))
                types_all.append(graph_attack_type(g))

            if embeddings and z is not None:
                graph_emb = scatter(z, batch.batch, dim=0, reduce="mean")
                embs_all.append(graph_emb.cpu().numpy())

            if components:
                comps["recon"].append(graph_mse.cpu())
                node_ce = F.cross_entropy(canid_logits, batch.node_id, reduction="none")
                comps["canid"].append(scatter(node_ce, batch.batch, reduce="mean").cpu())
                nbr_targets = vgae.create_neighborhood_targets(batch.node_id, batch.edge_index, batch.batch)
                node_nbr = F.binary_cross_entropy_with_logits(
                    nbr_logits, nbr_targets, reduction="none",
                ).mean(dim=1)
                comps["nbr"].append(scatter(node_nbr, batch.batch, reduce="mean").cpu())
                kl_val = kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
                n_graphs = int(batch.batch.max().item()) + 1
                comps["kl"].extend([kl_val] * n_graphs)

    return VGAEResult(
        errors=torch.cat(errors_all).numpy(),
        labels=np.array(labels_all),
        attack_types=np.array(types_all),
        embeddings=np.vstack(embs_all) if embs_all else None,
        components={k: (torch.cat(v).numpy() if isinstance(v[0], torch.Tensor) else np.array(v))
                    for k, v in comps.items()} if components else None,
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
    preds = (fused_scores > agent.decision_threshold).long()

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
