"""Evaluation stage: runs inference on validation and test data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphids.config import (
    PipelineConfig,
    cache_dir,
    data_dir,
    metrics_path,
    stage_dir,
)
from graphids.pipeline.artifacts import artifact_exists, get_artifact

from .data_loading import training_preamble
from .utils import (
    cache_predictions,
    cleanup,
    graph_label,
    load_frozen_cfg,
    load_model,
)

log = logging.getLogger(__name__)


def evaluate(cfg: PipelineConfig) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Output metrics.json layout:
        {
            "gat":    {"core": {...}, "additional": {...}},
            "vgae":   {"core": {...}, "additional": {...}},
            "fusion": {"core": {...}, "additional": {...}},
            "test": {
                "gat":    {"test_01_...": {"core": ...}, ...},
                "vgae":   {"test_01_...": {"core": ...}, ...},
                "fusion": {"test_01_...": {"core": ...}, ...}
            }
        }
    """
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "EVALUATION")
    test_scenarios = _load_test_data(cfg)

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}

    gat_stage = "curriculum"
    vgae_stage = "autoencoder"

    fusion_needs_models = (
        artifact_exists(cfg, "fusion", "best_model.pt", model_type="dqn")
        and artifact_exists(cfg, vgae_stage, "best_model.pt", model_type="vgae")
        and artifact_exists(cfg, gat_stage, "best_model.pt", model_type="gat")
    )

    # ---- Per-model evaluation ----
    gat, vgae = None, None

    if artifact_exists(cfg, gat_stage, "best_model.pt", model_type="gat"):
        gat = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)
        all_metrics["gat"], test_metrics["gat"] = _evaluate_gat(
            gat, val_data, test_scenarios, device
        )
        _collect_gat_artifacts(gat, val_data, device, artifacts)
        if not fusion_needs_models:
            del gat
            gat = None
            cleanup()

    if artifact_exists(cfg, vgae_stage, "best_model.pt", model_type="vgae"):
        vgae = load_model(cfg, "vgae", vgae_stage, num_ids, in_ch, device)
        all_metrics["vgae"], test_metrics["vgae"], best_thresh = _evaluate_vgae(
            vgae, val_data, test_scenarios, device
        )
        _collect_vgae_artifacts(vgae, val_data, device, artifacts)
        if not fusion_needs_models:
            del vgae
            vgae = None
            cleanup()

    if fusion_needs_models and vgae is not None and gat is not None:
        all_metrics["fusion"], test_metrics["fusion"] = _evaluate_fusion(
            cfg, gat, vgae, val_data, test_scenarios, device, artifacts
        )
        del vgae, gat
        cleanup()

    if cfg.temporal.enabled and artifact_exists(cfg, "temporal", "best_model.pt", model_type="gat"):
        temporal_m = _evaluate_temporal(cfg, val_data, num_ids, in_ch, device, gat_stage)
        if temporal_m is not None:
            all_metrics["temporal"] = temporal_m

    # ---- CKA (KD runs only) ----
    out = stage_dir(cfg, "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    if cfg.has_kd:
        try:
            _save_cka(cfg, val_data, device, num_ids, in_ch, out)
        except Exception as e:
            log.warning("CKA computation failed (non-fatal): %s", e)

    # ---- Persist ----
    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    mp = metrics_path(cfg, "evaluation")
    mp.write_text(json.dumps(all_metrics, indent=2))
    log.info("All metrics saved to %s", mp)

    _save_embedding_artifacts(artifacts, out)
    _save_attention_artifacts(artifacts, out)
    _save_dqn_policy_artifact(artifacts, out)

    cleanup()
    return all_metrics


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------


def _log_core_metrics(name: str, metrics: dict) -> None:
    log.info(
        "%s val metrics: %s",
        name,
        {k: f"{v:.4f}" for k, v in metrics["core"].items() if isinstance(v, float)},
    )


def _evaluate_gat(gat, val_data, test_scenarios, device) -> tuple[dict, dict]:
    """Evaluate GAT on validation + test scenarios. Returns (val_metrics, test_metrics)."""
    p, l, s, _, _, _ = _run_gat_inference(gat, val_data, device)
    val_m = _compute_metrics(l, p, s)
    _log_core_metrics("GAT", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            tp, tl, ts, _, _, _ = _run_gat_inference(gat, tdata, device)
            test_m[scenario] = _compute_metrics(tl, tp, ts)
            log.info(
                "GAT %s  acc=%.4f f1=%.4f",
                scenario,
                test_m[scenario]["core"]["accuracy"],
                test_m[scenario]["core"]["f1"],
            )
    return val_m, test_m


def _collect_gat_artifacts(gat, val_data, device, artifacts: dict) -> None:
    """Run GAT inference with embedding + attention capture for artifact export."""
    p, l, s, gat_emb, gat_attn, gat_at = _run_gat_inference(
        gat, val_data, device, capture_embeddings=True, capture_attention=True
    )
    if gat_emb is not None:
        artifacts["gat_emb"] = gat_emb
        artifacts["gat_labels"] = l
        artifacts["gat_attack_types"] = gat_at
    if gat_attn:
        artifacts["gat_attention"] = gat_attn


def _evaluate_vgae(vgae, val_data, test_scenarios, device) -> tuple[dict, dict, float]:
    """Evaluate VGAE on validation + test. Returns (val_metrics, test_metrics, threshold)."""
    errors_np, labels_np, _, _, _ = _run_vgae_inference(vgae, val_data, device)
    best_thresh, youden_j, vgae_preds = _vgae_threshold(labels_np, errors_np)
    val_m = _compute_metrics(labels_np, vgae_preds, errors_np)
    val_m["core"]["optimal_threshold"] = best_thresh
    val_m["core"]["youden_j"] = youden_j
    _log_core_metrics("VGAE", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            te, tl, _, _, _ = _run_vgae_inference(vgae, tdata, device)
            tp = (te > best_thresh).astype(int)
            test_m[scenario] = _compute_metrics(tl, tp, te)
            test_m[scenario]["core"]["threshold_from_val"] = best_thresh
            log.info(
                "VGAE %s  acc=%.4f f1=%.4f",
                scenario,
                test_m[scenario]["core"]["accuracy"],
                test_m[scenario]["core"]["f1"],
            )
    return val_m, test_m, best_thresh


def _collect_vgae_artifacts(vgae, val_data, device, artifacts: dict) -> None:
    """Run VGAE inference with embedding + component capture for artifact export."""
    errors_np, labels_np, vgae_z, vgae_at, vgae_components = _run_vgae_inference(
        vgae, val_data, device, capture_embeddings=True, capture_components=True
    )
    if vgae_z is not None:
        artifacts["vgae_z"] = vgae_z
        artifacts["vgae_labels"] = labels_np
        artifacts["vgae_errors"] = errors_np
        artifacts["vgae_attack_types"] = vgae_at
    if vgae_components is not None:
        for comp_name, comp_arr in vgae_components.items():
            artifacts[f"vgae_error_{comp_name}"] = comp_arr


def _evaluate_fusion(
    cfg, gat, vgae, val_data, test_scenarios, device, artifacts
) -> tuple[dict, dict]:
    """Evaluate DQN/MLP/WeightedAvg fusion. Returns (val_metrics, test_metrics)."""
    models = {"vgae": vgae, "gat": gat}
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)

    from graphids.core.models.dqn import EnhancedDQNFusionAgent

    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    fusion_ckpt = get_artifact(cfg, "fusion", "best_model.pt", model_type="dqn")
    agent = EnhancedDQNFusionAgent.from_config(fusion_cfg, device=str(device), inference=True)
    agent.load_checkpoint(fusion_ckpt)

    fp, fl, fs, fq = _run_fusion_inference(agent, val_cache)
    val_m = _compute_metrics(fl, fp, fs)
    artifacts["dqn_alphas"] = fs.tolist()
    artifacts["dqn_labels"] = fl.tolist()
    artifacts["dqn_q_values"] = fq
    _log_core_metrics("Fusion", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
            tp, tl, ts, _ = _run_fusion_inference(agent, tc)
            test_m[scenario] = _compute_metrics(tl, tp, ts)
            log.info(
                "Fusion %s  acc=%.4f f1=%.4f",
                scenario,
                test_m[scenario]["core"]["accuracy"],
                test_m[scenario]["core"]["f1"],
            )
    return val_m, test_m


def _evaluate_temporal(cfg, val_data, num_ids, in_ch, device, gat_stage) -> dict | None:
    """Evaluate temporal model. Returns metrics dict or None on failure."""
    try:
        from graphids.core.models.temporal import TemporalGraphClassifier
        from graphids.core.preprocessing._temporal import TemporalGrouper

        gat_for_temporal = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)
        spatial_dim = probe_embedding_dim(gat_for_temporal, val_data[0], device)

        tc = cfg.temporal
        temporal_model = TemporalGraphClassifier(
            spatial_encoder=gat_for_temporal,
            spatial_dim=spatial_dim,
            temporal_hidden=tc.temporal_hidden,
            temporal_heads=tc.temporal_heads,
            temporal_layers=tc.temporal_layers,
            max_seq_len=tc.temporal_window,
            freeze_spatial=True,
            num_classes=2,
        ).to(device)
        temporal_ckpt = get_artifact(cfg, "temporal", "best_model.pt", model_type="gat")
        temporal_model.load_state_dict(
            torch.load(temporal_ckpt, map_location="cpu", weights_only=True)
        )
        temporal_model.eval()

        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        val_sequences = grouper.group(val_data)

        if not val_sequences:
            return None

        t_preds, t_labels = [], []
        with torch.no_grad():
            for seq_obj in val_sequences:
                moved = [g.clone().to(device) for g in seq_obj.graphs]
                logits = temporal_model([[g for g in moved]])
                t_preds.append(logits.argmax(dim=1)[0].item())
                t_labels.append(seq_obj.y)

        metrics = _compute_metrics(np.array(t_labels), np.array(t_preds))
        _log_core_metrics("Temporal", metrics)

        del temporal_model, gat_for_temporal
        cleanup()
        return metrics
    except Exception as e:
        log.warning("Temporal evaluation failed (non-fatal): %s", e)
        return None


def probe_embedding_dim(model, sample_graph, device) -> int:
    """Probe a model's embedding dimension using a single forward pass."""
    with torch.no_grad():
        probe = sample_graph.clone().to(device)
        _, emb = model(probe, return_embedding=True)
        return emb.shape[-1]


# ---------------------------------------------------------------------------
# Artifact saving helpers
# ---------------------------------------------------------------------------


def _save_embedding_artifacts(artifacts: dict, out: Path) -> None:
    embed_data = {}
    for key in (
        "vgae_z",
        "gat_emb",
        "vgae_labels",
        "gat_labels",
        "vgae_errors",
        "vgae_error_recon",
        "vgae_error_canid",
        "vgae_error_nbr",
        "vgae_error_kl",
        "vgae_attack_types",
        "gat_attack_types",
    ):
        if key in artifacts:
            embed_data[key] = artifacts[key]
    if embed_data:
        npz_path = out / "embeddings.npz"
        np.savez_compressed(npz_path, **embed_data)
        log.info("Saved embeddings → %s", npz_path)


def _save_attention_artifacts(artifacts: dict, out: Path) -> None:
    if "gat_attention" not in artifacts:
        return
    attn_list = artifacts["gat_attention"]
    attn_export = {}
    for i, entry in enumerate(attn_list):
        prefix = f"sample_{i}"
        attn_export[f"{prefix}_graph_idx"] = entry["graph_idx"]
        attn_export[f"{prefix}_label"] = entry["label"]
        attn_export[f"{prefix}_edge_index"] = entry["edge_index"]
        attn_export[f"{prefix}_node_features"] = entry["node_features"]
        for layer_idx, aw in enumerate(entry["attention_weights"]):
            attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = aw
    attn_export["n_samples"] = len(attn_list)
    attn_path = out / "attention_weights.npz"
    np.savez_compressed(attn_path, **attn_export)
    log.info("Saved attention weights (%d samples) → %s", len(attn_list), attn_path)


def _save_dqn_policy_artifact(artifacts: dict, out: Path) -> None:
    if "dqn_alphas" not in artifacts:
        return
    alphas = artifacts["dqn_alphas"]
    labels = artifacts["dqn_labels"]
    alpha_by_label = {"normal": [], "attack": []}
    for a, lbl in zip(alphas, labels):
        alpha_by_label["normal" if lbl == 0 else "attack"].append(a)
    policy_data = {
        "alphas": alphas,
        "labels": labels,
        "alpha_by_label": alpha_by_label,
        "q_values": artifacts.get("dqn_q_values", []),
    }
    policy_path = out / "dqn_policy.json"
    policy_path.write_text(json.dumps(policy_data, indent=2))
    log.info("Saved DQN policy → %s", policy_path)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_test_data(cfg: PipelineConfig) -> dict:
    """Load held-out test graphs per scenario (cached)."""
    from graphids.core.preprocessing import PreprocessingPipeline

    return PreprocessingPipeline(cfg).load_test_scenarios()


ATTENTION_SAMPLE_LIMIT = 50  # Max graphs to capture attention for (export size)


def _run_gat_inference(gat, data, device, capture_embeddings=False, capture_attention=False):
    """Run GAT inference. Returns (preds, labels, scores, embeddings, attn_data, attack_types).

    Uses batched inference via PyG DataLoader for 10-50x speedup.
    Attention capture stays per-sample (small subset only).
    """
    from torch_geometric.loader import DataLoader as PyGDataLoader

    from graphids.core.preprocessing import graph_attack_type

    preds, labels, scores = [], [], []
    attack_types = []
    embeddings = [] if capture_embeddings else None
    attn_data = [] if capture_attention else None

    loader = PyGDataLoader(data, batch_size=128, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if capture_embeddings:
                logits, emb = gat(batch, return_embedding=True)
                for row in emb.cpu().numpy():
                    embeddings.append(row)
            else:
                logits = gat(batch)
            probs = F.softmax(logits, dim=1)
            preds.extend(logits.argmax(1).cpu().tolist())
            scores.extend(probs[:, 1].cpu().tolist())
            labels.extend(batch.y.cpu().tolist())
            if hasattr(batch, "attack_type") and batch.attack_type is not None:
                attack_types.extend(batch.attack_type.cpu().tolist())
            else:
                attack_types.extend([-1] * batch.num_graphs)

    # Attention capture (separate per-sample pass, small subset only)
    if capture_attention:
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

    emb_array = np.array(embeddings) if capture_embeddings and embeddings else None
    return (
        np.array(preds),
        np.array(labels),
        np.array(scores),
        emb_array,
        attn_data,
        np.array(attack_types),
    )


def _run_vgae_inference(vgae, data, device, capture_embeddings=False, capture_components=False):
    """Run VGAE reconstruction-error inference.

    Returns (errors, labels, embeddings, attack_types, component_errors).

    Uses batched inference via PyG DataLoader for the common path.
    Falls back to per-sample when capture_components=True (needs per-graph
    neighborhood targets and KL decomposition).
    """
    if capture_components:
        return _run_vgae_inference_per_sample(
            vgae, data, device, capture_embeddings, capture_components
        )

    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import global_mean_pool
    from torch_geometric.utils import scatter

    errors, labels = [], []
    attack_types = []
    embeddings = [] if capture_embeddings else None

    loader = PyGDataLoader(data, batch_size=128, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = getattr(batch, "edge_attr", None)
            cont, _, _, z, _ = vgae(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)

            # Per-graph MSE via scatter
            per_node_se = (cont - batch.x[:, 1:]).pow(2).mean(dim=1)
            graph_errors = scatter(per_node_se, batch.batch, dim=0, reduce="mean")
            errors.extend(graph_errors.cpu().tolist())

            labels.extend(batch.y.cpu().tolist())
            if hasattr(batch, "attack_type") and batch.attack_type is not None:
                attack_types.extend(batch.attack_type.cpu().tolist())
            else:
                attack_types.extend([-1] * batch.num_graphs)

            if capture_embeddings and z is not None:
                graph_z = global_mean_pool(z, batch.batch)
                for row in graph_z.cpu().numpy():
                    embeddings.append(row)

    emb_array = np.array(embeddings) if capture_embeddings and embeddings else None
    return np.array(errors), np.array(labels), emb_array, np.array(attack_types), None


def _run_vgae_inference_per_sample(vgae, data, device, capture_embeddings, capture_components):
    """Per-sample VGAE inference with component-level loss decomposition."""
    from graphids.core.preprocessing import get_batch_index, graph_attack_type

    errors, labels = [], []
    attack_types = []
    embeddings = [] if capture_embeddings else None
    components = {"recon": [], "canid": [], "nbr": [], "kl": []} if capture_components else None
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
            if capture_components:
                components["recon"].append(err)
                components["canid"].append(F.cross_entropy(canid_logits, g.x[:, 0].long()).item())
                nbr_targets = vgae.create_neighborhood_targets(g.x, g.edge_index, batch_idx)
                components["nbr"].append(
                    F.binary_cross_entropy_with_logits(nbr_logits, nbr_targets).item()
                )
                components["kl"].append(
                    kl_loss.item() if torch.is_tensor(kl_loss) else float(kl_loss)
                )
    emb_array = np.array(embeddings) if capture_embeddings and embeddings else None
    comp_arrays = {k: np.array(v) for k, v in components.items()} if capture_components else None
    return np.array(errors), np.array(labels), emb_array, np.array(attack_types), comp_arrays


def _run_fusion_inference(agent, cache):
    """Run DQN fusion inference (vectorized). Returns (preds, labels, scores, q_values_list)."""
    states = cache["states"]  # [N, D] tensor
    labels_t = cache["labels"]  # [N] tensor

    actions, alphas, norm_states = agent.select_action_batch(states, training=False)
    anomaly_scores, gat_probs = agent._derive_scores_batch(norm_states)
    fused_scores = (1 - alphas) * anomaly_scores + alphas * gat_probs
    preds = (fused_scores > 0.5).long()

    with torch.no_grad():
        q_values = agent.q_network(norm_states.to(agent.device)).cpu()

    return (
        preds.numpy(),
        labels_t.numpy(),
        fused_scores.numpy(),
        q_values.numpy().tolist(),
    )


def _vgae_threshold(labels, errors):
    """Find optimal anomaly-detection threshold via Youden's J statistic."""
    from sklearn.metrics import roc_curve as _roc_curve

    fpr_v, tpr_v, thresholds_v = _roc_curve(labels, errors)
    j_scores = tpr_v - fpr_v
    best_idx = np.argmax(j_scores)
    best_thresh = (
        float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(np.median(errors))
    )
    preds = (errors > best_thresh).astype(int)
    return best_thresh, float(j_scores[best_idx]), preds


def _compute_metrics(labels, preds, scores=None) -> dict:
    """Compute classification metrics using torchmetrics.

    Core metrics via MetricCollection (GPU-native, no sklearn).
    Custom: detection-at-FPR thresholds (no torchmetrics equivalent).
    """
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryCohenKappa,
        BinaryConfusionMatrix,
        BinaryF1Score,
        BinaryMatthewsCorrCoef,
        BinaryPrecision,
        BinaryRecall,
        BinaryROC,
        BinarySpecificity,
    )

    labels_t = torch.as_tensor(labels, dtype=torch.long)
    preds_t = torch.as_tensor(preds, dtype=torch.long)

    # Confusion matrix first — used for FPR/FNR and per-class stats
    cm = BinaryConfusionMatrix()(preds_t, labels_t)
    tn, fp, fn, tp = cm.ravel().tolist()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    precision = BinaryPrecision()(preds_t, labels_t).item()
    recall = BinaryRecall()(preds_t, labels_t).item()
    specificity = BinarySpecificity()(preds_t, labels_t).item()

    core = {
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "precision": precision,
        "recall": recall,
        "f1": BinaryF1Score()(preds_t, labels_t).item(),
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": BinaryMatthewsCorrCoef()(preds_t, labels_t).item(),
        "fpr": fpr,
        "fnr": fnr,
        "n_samples": int(len(labels)),
        "confusion_matrix": cm.tolist(),
    }

    # Per-class precision / recall / f1 / support
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    prec_1, rec_1, f1_1 = precision, recall, core["f1"]

    additional = {
        "kappa": BinaryCohenKappa()(preds_t, labels_t).item(),
        "per_class": {
            "0": {"precision": prec_0, "recall": rec_0, "f1-score": f1_0, "support": int(tn + fp)},
            "1": {"precision": prec_1, "recall": rec_1, "f1-score": f1_1, "support": int(tp + fn)},
        },
    }

    if scores is not None and len(set(labels)) > 1:
        scores_t = torch.as_tensor(scores, dtype=torch.float)
        core["auc"] = BinaryAUROC()(scores_t, labels_t).item()
        try:
            additional["pr_auc"] = BinaryAveragePrecision()(scores_t, labels_t).item()
        except ValueError:
            additional["pr_auc"] = 0.0
        try:
            fpr_curve, tpr_curve, _ = BinaryROC()(scores_t, labels_t)
            det_at_fpr = {}
            for fpr_target in [0.05, 0.01, 0.001]:
                idx = torch.argmin(torch.abs(fpr_curve - fpr_target))
                det_at_fpr[str(fpr_target)] = float(tpr_curve[idx])
            additional["detection_at_fpr"] = det_at_fpr
        except ValueError:
            additional["detection_at_fpr"] = {}

    return {"core": core, "additional": additional}


# ---------------------------------------------------------------------------
# CKA (Centered Kernel Alignment) for KD transfer analysis
# ---------------------------------------------------------------------------


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two representation matrices.

    Args:
        X: [n_samples, dim_x] representation matrix.
        Y: [n_samples, dim_y] representation matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    n = X.shape[0]

    hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2 / (n - 1) ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2 / (n - 1) ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2 / (n - 1) ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 0 else 0.0


def _collect_layer_representations(model, data, device, max_samples=500):
    """Collect per-layer representations from a GAT model."""
    all_layers = None
    count = 0
    with torch.no_grad():
        for g in data:
            if count >= max_samples:
                break
            g = g.clone().to(device)
            xs = model(g, return_intermediate=True)
            # Mean-pool each layer over nodes → graph-level representation
            layer_reps = [x.mean(dim=0).cpu().numpy() for x in xs]
            if all_layers is None:
                all_layers = [[] for _ in range(len(layer_reps))]
            for i, rep in enumerate(layer_reps):
                all_layers[i].append(rep)
            count += 1
    if all_layers is None:
        return []
    return [np.array(layer) for layer in all_layers]


def _save_cka(cfg, val_data, device, num_ids, in_ch, out_dir):
    """Compute and save CKA matrix between teacher and student GAT layers."""
    from graphids.config import resolve
    from graphids.pipeline.artifacts import artifact_exists

    teacher_cfg = resolve("gat", "large", dataset=cfg.dataset)
    if not artifact_exists(teacher_cfg, "curriculum", "best_model.pt", model_type="gat"):
        log.warning("CKA: teacher checkpoint not found")
        return

    if not artifact_exists(cfg, "curriculum", "best_model.pt", model_type="gat"):
        log.warning("CKA: student checkpoint not found")
        return

    teacher = load_model(teacher_cfg, "gat", "curriculum", num_ids, in_ch, device)
    student = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)

    teacher_layers = _collect_layer_representations(teacher, val_data, device)
    student_layers = _collect_layer_representations(student, val_data, device)

    if not teacher_layers or not student_layers:
        log.warning("CKA: empty layer representations")
        return

    # Compute CKA matrix: rows=teacher layers, cols=student layers
    n_teacher = len(teacher_layers)
    n_student = len(student_layers)
    cka_matrix = np.zeros((n_teacher, n_student))
    for i in range(n_teacher):
        for j in range(n_student):
            cka_matrix[i, j] = _linear_cka(teacher_layers[i], student_layers[j])

    cka_data = {
        "matrix": cka_matrix.tolist(),
        "teacher_layers": [f"Teacher L{i + 1}" for i in range(n_teacher)],
        "student_layers": [f"Student L{i + 1}" for i in range(n_student)],
    }
    cka_path = out_dir / "cka_matrix.json"
    cka_path.write_text(json.dumps(cka_data, indent=2))
    log.info("Saved CKA matrix (%dx%d) → %s", n_teacher, n_student, cka_path)

    del teacher, student
    cleanup()
