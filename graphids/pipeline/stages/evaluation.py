"""Evaluation stage: runs inference on validation and test data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graphids.config import PipelineConfig, cache_dir, data_dir, metrics_path, stage_dir
from graphids.config.constants import get_batch_index, graph_attack_type

from .data_loading import training_preamble
from .utils import (
    _cross_model_path,
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

    # Stage names (run_id() adds aux suffix automatically based on cfg.auxiliaries)
    gat_stage = "curriculum"
    vgae_stage = "autoencoder"

    # Artifact collectors for embeddings/policy export
    artifacts: dict = {}

    # ---- GAT evaluation ----
    gat_ckpt = _cross_model_path(cfg, "gat", gat_stage, "best_model.pt")
    if gat_ckpt.exists():
        gat = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)

        p, l, s, gat_emb, gat_attn, gat_at = _run_gat_inference(
            gat,
            val_data,
            device,
            capture_embeddings=True,
            capture_attention=True,
        )
        all_metrics["gat"] = _compute_metrics(l, p, s)
        if gat_emb is not None:
            artifacts["gat_emb"] = gat_emb
            artifacts["gat_labels"] = l
            artifacts["gat_attack_types"] = gat_at
        if gat_attn:
            artifacts["gat_attention"] = gat_attn
        log.info(
            "GAT val metrics: %s",
            {k: f"{v:.4f}" for k, v in all_metrics["gat"]["core"].items() if isinstance(v, float)},
        )

        if test_scenarios:
            test_metrics["gat"] = {}
            for scenario, tdata in test_scenarios.items():
                tp, tl, ts, _, _, _ = _run_gat_inference(gat, tdata, device)
                test_metrics["gat"][scenario] = _compute_metrics(tl, tp, ts)
                log.info(
                    "GAT %s  acc=%.4f f1=%.4f",
                    scenario,
                    test_metrics["gat"][scenario]["core"]["accuracy"],
                    test_metrics["gat"][scenario]["core"]["f1"],
                )

        del gat
        cleanup()

    # ---- VGAE evaluation ----
    vgae_ckpt = _cross_model_path(cfg, "vgae", vgae_stage, "best_model.pt")
    if vgae_ckpt.exists():
        vgae = load_model(cfg, "vgae", vgae_stage, num_ids, in_ch, device)

        errors_np, labels_np, vgae_z, vgae_at = _run_vgae_inference(
            vgae, val_data, device, capture_embeddings=True
        )
        best_thresh, youden_j, vgae_preds = _vgae_threshold(labels_np, errors_np)
        all_metrics["vgae"] = _compute_metrics(labels_np, vgae_preds, errors_np)
        if vgae_z is not None:
            artifacts["vgae_z"] = vgae_z
            artifacts["vgae_labels"] = labels_np
            artifacts["vgae_errors"] = errors_np
            artifacts["vgae_attack_types"] = vgae_at
        all_metrics["vgae"]["core"]["optimal_threshold"] = best_thresh
        all_metrics["vgae"]["core"]["youden_j"] = youden_j
        log.info(
            "VGAE val metrics: %s",
            {k: f"{v:.4f}" for k, v in all_metrics["vgae"]["core"].items() if isinstance(v, float)},
        )

        if test_scenarios:
            test_metrics["vgae"] = {}
            for scenario, tdata in test_scenarios.items():
                te, tl, _, _ = _run_vgae_inference(vgae, tdata, device)
                tp = (te > best_thresh).astype(int)
                test_metrics["vgae"][scenario] = _compute_metrics(tl, tp, te)
                test_metrics["vgae"][scenario]["core"]["threshold_from_val"] = best_thresh
                log.info(
                    "VGAE %s  acc=%.4f f1=%.4f",
                    scenario,
                    test_metrics["vgae"][scenario]["core"]["accuracy"],
                    test_metrics["vgae"][scenario]["core"]["f1"],
                )

        del vgae
        cleanup()

    # ---- DQN Fusion evaluation ----
    fusion_ckpt = _cross_model_path(cfg, "dqn", "fusion", "best_model.pt")
    if fusion_ckpt.exists() and vgae_ckpt.exists() and gat_ckpt.exists():
        vgae = load_model(cfg, "vgae", vgae_stage, num_ids, in_ch, device)
        gat = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)

        models = {"vgae": vgae, "gat": gat}
        val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)

        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        from graphids.core.models.registry import fusion_state_dim

        fusion_cfg = load_frozen_cfg(cfg, "fusion")
        agent = EnhancedDQNFusionAgent(
            lr=fusion_cfg.fusion.lr,
            gamma=fusion_cfg.dqn.gamma,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            buffer_size=fusion_cfg.dqn.buffer_size,
            batch_size=fusion_cfg.dqn.batch_size,
            target_update_freq=fusion_cfg.dqn.target_update,
            device=str(device),
            state_dim=fusion_state_dim(),
            alpha_steps=fusion_cfg.fusion.alpha_steps,
            hidden_dim=fusion_cfg.dqn.hidden,
            num_layers=fusion_cfg.dqn.layers,
        )
        fusion_sd = torch.load(fusion_ckpt, map_location="cpu", weights_only=True)
        agent.q_network.load_state_dict(fusion_sd["q_network"])
        agent.target_network.load_state_dict(fusion_sd["target_network"])

        fp, fl, fs, fq = _run_fusion_inference(agent, val_cache)
        all_metrics["fusion"] = _compute_metrics(fl, fp, fs)
        # Capture DQN policy (alpha distribution by class)
        artifacts["dqn_alphas"] = fs.tolist()
        artifacts["dqn_labels"] = fl.tolist()
        artifacts["dqn_q_values"] = fq
        log.info(
            "Fusion val metrics: %s",
            {
                k: f"{v:.4f}"
                for k, v in all_metrics["fusion"]["core"].items()
                if isinstance(v, float)
            },
        )

        if test_scenarios:
            test_metrics["fusion"] = {}
            for scenario, tdata in test_scenarios.items():
                tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
                tp, tl, ts, _ = _run_fusion_inference(agent, tc)
                test_metrics["fusion"][scenario] = _compute_metrics(tl, tp, ts)
                log.info(
                    "Fusion %s  acc=%.4f f1=%.4f",
                    scenario,
                    test_metrics["fusion"][scenario]["core"]["accuracy"],
                    test_metrics["fusion"][scenario]["core"]["f1"],
                )

        del vgae, gat
        cleanup()

    if test_metrics:
        all_metrics["test"] = test_metrics

    # Save all metrics
    out = stage_dir(cfg, "evaluation")
    out.mkdir(parents=True, exist_ok=True)
    mp = metrics_path(cfg, "evaluation")
    mp.write_text(json.dumps(all_metrics, indent=2))
    log.info("All metrics saved to %s", mp)

    # Save embeddings artifact (VGAE latent + GAT hidden + attack types)
    embed_data = {}
    for key in (
        "vgae_z",
        "gat_emb",
        "vgae_labels",
        "gat_labels",
        "vgae_errors",
        "vgae_attack_types",
        "gat_attack_types",
    ):
        if key in artifacts:
            embed_data[key] = artifacts[key]
    if embed_data:
        npz_path = out / "embeddings.npz"
        np.savez_compressed(npz_path, **embed_data)
        log.info("Saved embeddings → %s", npz_path)

    # Save attention weights artifact
    if "gat_attention" in artifacts:
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

    # Temporal model evaluation
    if cfg.temporal.enabled:
        temporal_ckpt = _cross_model_path(cfg, "gat", "temporal", "best_model.pt")
        if temporal_ckpt.exists():
            try:
                from graphids.core.models.temporal import TemporalGraphClassifier
                from graphids.core.preprocessing.temporal import TemporalGrouper

                # Load spatial encoder
                gat_for_temporal = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)

                # Probe spatial dim
                with torch.no_grad():
                    probe = val_data[0].clone().to(device)
                    _, probe_emb = gat_for_temporal(probe, return_embedding=True)
                    spatial_dim = probe_emb.shape[-1]

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
                temporal_model.load_state_dict(
                    torch.load(temporal_ckpt, map_location="cpu", weights_only=True)
                )
                temporal_model.eval()

                grouper = TemporalGrouper(
                    window=tc.temporal_window,
                    stride=tc.temporal_stride,
                )
                val_sequences = grouper.group(val_data)

                if val_sequences:
                    t_preds, t_labels = [], []
                    with torch.no_grad():
                        for seq_obj in val_sequences:
                            moved = [g.clone().to(device) for g in seq_obj.graphs]
                            logits = temporal_model([[g for g in moved]])
                            t_preds.append(logits.argmax(dim=1)[0].item())
                            t_labels.append(seq_obj.y)

                    all_metrics["temporal"] = _compute_metrics(
                        np.array(t_labels),
                        np.array(t_preds),
                    )
                    log.info(
                        "Temporal val metrics: %s",
                        {
                            k: f"{v:.4f}"
                            for k, v in all_metrics["temporal"]["core"].items()
                            if isinstance(v, float)
                        },
                    )

                del temporal_model, gat_for_temporal
                cleanup()
            except Exception as e:
                log.warning("Temporal evaluation failed (non-fatal): %s", e)

    # GNNExplainer feature importance
    if cfg.training.run_explainer:
        gat_ckpt_for_explain = _cross_model_path(cfg, "gat", gat_stage, "best_model.pt")
        if gat_ckpt_for_explain.exists():
            try:
                from graphids.core.explain import explain_graphs

                gat_for_explain = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)
                explanations = explain_graphs(
                    gat_for_explain,
                    "gat",
                    val_data[: cfg.training.explainer_samples],
                    device,
                    n_samples=cfg.training.explainer_samples,
                    epochs=cfg.training.explainer_epochs,
                )
                np.savez_compressed(out / "explanations.npz", **explanations)
                log.info("Saved explanations for %d graphs", len(explanations["graph_indices"]))
                del gat_for_explain
                cleanup()
            except Exception as e:
                log.warning("GNNExplainer failed (non-fatal): %s", e)

    # CKA computation for KD runs (teacher vs student layer similarity)
    if cfg.has_kd:
        try:
            _save_cka(cfg, val_data, device, num_ids, in_ch, out)
        except Exception as e:
            log.warning("CKA computation failed (non-fatal): %s", e)

    # Save DQN policy artifact
    if "dqn_alphas" in artifacts:
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

    cleanup()
    return all_metrics


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_test_data(cfg: PipelineConfig) -> dict:
    """Load held-out test graphs per scenario (cached)."""
    from graphids.core.training.datamodules import load_test_scenarios

    return load_test_scenarios(
        cfg.dataset,
        data_dir(cfg),
        cache_dir(cfg),
    )


ATTENTION_SAMPLE_LIMIT = 50  # Max graphs to capture attention for (export size)


def _run_gat_inference(gat, data, device, capture_embeddings=False, capture_attention=False):
    """Run GAT inference. Returns (preds, labels, scores, embeddings, attn_data, attack_types).

    When capture_embeddings=True, captures the hidden representation before
    the final classification layer via the forward_embedding() method.
    When capture_attention=True, captures per-layer attention weights for a
    sampled subset of graphs.
    """
    preds, labels, scores = [], [], []
    attack_types = []
    embeddings = [] if capture_embeddings else None
    attn_data = [] if capture_attention else None
    with torch.no_grad():
        for idx, g in enumerate(data):
            g = g.clone().to(device)
            if capture_embeddings:
                logits, emb = gat(g, return_embedding=True)
                embeddings.append(emb[0].cpu().numpy())
            else:
                logits = gat(g)
            probs = F.softmax(logits, dim=1)
            preds.append(logits.argmax(1)[0].item())
            labels.append(graph_label(g))
            scores.append(probs[0, 1].item())
            attack_types.append(graph_attack_type(g))
            # Attention capture (separate pass, sampled subset only)
            if capture_attention and idx < ATTENTION_SAMPLE_LIMIT:
                _, att_weights = gat(g, return_attention_weights=True)
                attn_data.append(
                    {
                        "graph_idx": idx,
                        "label": graph_label(g),
                        "edge_index": g.edge_index.cpu().numpy(),
                        "node_features": g.x[:, 0].cpu().numpy(),  # CAN IDs
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


def _run_vgae_inference(vgae, data, device, capture_embeddings=False):
    """Run VGAE reconstruction-error inference. Returns (errors, labels, embeddings, attack_types).

    When capture_embeddings=True, captures z.mean(dim=0) (graph-level latent
    embedding) per sample from the encoder's latent representation.
    """
    errors, labels = [], []
    attack_types = []
    embeddings = [] if capture_embeddings else None
    with torch.no_grad():
        for g in data:
            g = g.clone().to(device)
            batch_idx = get_batch_index(g, device)
            edge_attr = getattr(g, "edge_attr", None)
            cont, canid_logits, z_mean, z_logstd, _ = vgae(
                g.x, g.edge_index, batch_idx, edge_attr=edge_attr
            )
            err = F.mse_loss(cont, g.x[:, 1:]).item()
            errors.append(err)
            labels.append(graph_label(g))
            attack_types.append(graph_attack_type(g))
            if capture_embeddings and z_mean is not None:
                # Graph-level embedding: mean pool over nodes
                embeddings.append(z_mean.mean(dim=0).cpu().numpy())
    emb_array = np.array(embeddings) if capture_embeddings and embeddings else None
    return np.array(errors), np.array(labels), emb_array, np.array(attack_types)


def _run_fusion_inference(agent, cache):
    """Run DQN fusion inference. Returns (preds, labels, scores, q_values_list)."""
    preds, labels, scores, q_values_list = [], [], [], []
    for i in range(len(cache["states"])):
        state_np = cache["states"][i].numpy()
        # Capture raw Q-values before action selection
        state_t = (
            torch.tensor(agent.normalize_state(state_np), dtype=torch.float32)
            .unsqueeze(0)
            .to(agent.device)
        )
        with torch.no_grad():
            q_vals = agent.q_network(state_t).squeeze(0).cpu().numpy()
        q_values_list.append(q_vals.tolist())
        alpha, _, _ = agent.select_action(state_np, training=False)
        preds.append(1 if alpha > 0.5 else 0)
        labels.append(cache["labels"][i].item())
        scores.append(float(alpha))
    return np.array(preds), np.array(labels), np.array(scores), q_values_list


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
    """Compute comprehensive classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.metrics import (
        auc as sk_auc,
    )

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    core = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "fpr": fpr,
        "fnr": fnr,
        "n_samples": int(len(labels)),
        "confusion_matrix": cm.tolist(),
    }

    additional = {
        "kappa": float(cohen_kappa_score(labels, preds)),
        "tpr": tpr,
        "tnr": specificity,
        "detection_rate": tpr,
        "miss_rate": fnr,
    }

    if scores is not None and len(set(labels)) > 1:
        core["auc"] = float(roc_auc_score(labels, scores))

        try:
            prec_vals, rec_vals, _ = precision_recall_curve(labels, scores)
            additional["pr_auc"] = float(sk_auc(rec_vals, prec_vals))
            # Downsample PR curve for export
            step = max(1, len(prec_vals) // 200)
            additional["pr_curve"] = {
                "precision": prec_vals[::step].tolist(),
                "recall": rec_vals[::step].tolist(),
            }
        except ValueError:
            additional["pr_auc"] = 0.0

        try:
            fpr_curve, tpr_curve, _ = roc_curve(labels, scores)
            det_at_fpr = {}
            for fpr_target in [0.05, 0.01, 0.001]:
                idx = np.argmin(np.abs(fpr_curve - fpr_target))
                det_at_fpr[str(fpr_target)] = float(tpr_curve[idx])
            additional["detection_at_fpr"] = det_at_fpr
            # Downsample ROC curve for export
            step = max(1, len(fpr_curve) // 200)
            additional["roc_curve"] = {
                "fpr": fpr_curve[::step].tolist(),
                "tpr": tpr_curve[::step].tolist(),
            }
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
    from graphids.config import stage_dir

    # Teacher is large-scale, same dataset, no KD
    teacher_gat_stage = "curriculum"
    teacher_ckpt = _cross_model_path(cfg, "gat", teacher_gat_stage, "best_model.pt")

    # For KD runs, we need the teacher (large, no-KD) model
    # The teacher path is the large-scale GAT without KD auxiliary
    from graphids.config import checkpoint_path, resolve

    teacher_cfg = resolve("gat", "large", dataset=cfg.dataset)
    teacher_ckpt = checkpoint_path(teacher_cfg, teacher_gat_stage)

    if not teacher_ckpt.exists():
        log.warning("CKA: teacher checkpoint not found at %s", teacher_ckpt)
        return

    student_ckpt = _cross_model_path(cfg, "gat", "curriculum", "best_model.pt")
    if not student_ckpt.exists():
        log.warning("CKA: student checkpoint not found at %s", student_ckpt)
        return

    teacher = load_model(teacher_cfg, "gat", teacher_gat_stage, num_ids, in_ch, device)
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
