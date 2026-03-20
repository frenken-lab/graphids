"""Evaluation stage: runs inference on validation and test data."""

from __future__ import annotations

import structlog

import numpy as np
import torch

from pathlib import Path

from graphids.config import PipelineConfig

from .data_loading import cache_predictions, cleanup, training_preamble
from .eval_inference import run_fusion_inference, run_gat_inference, run_vgae_inference
from .trainer_factory import load_frozen_cfg, load_model

log = structlog.get_logger()


def evaluate(cfg: PipelineConfig) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Returns ``{"metrics": {...}}`` where metrics has the structure::

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

    cli.py passes this to the manifest (single source of truth for metrics).
    """
    train_data, val_data, num_ids, in_ch, device = training_preamble(cfg, "EVALUATION")
    test_scenarios = _load_test_data(cfg)

    all_metrics: dict = {}
    test_metrics: dict = {}

    fusion_needs_models = (
        Path(cfg.checkpoints["dqn"]).exists()
        and Path(cfg.checkpoints["vgae"]).exists()
        and Path(cfg.checkpoints["gat"]).exists()
    )

    # ---- Per-model evaluation ----
    gat_model, vgae_model = None, None
    gat_result, vgae_result, fusion_result = None, None, None

    if Path(cfg.checkpoints["gat"]).exists():
        gat_model = load_model(cfg, "gat", gat_stage, num_ids, in_ch, device)
        all_metrics["gat"], test_metrics["gat"] = _evaluate_gat(
            gat_model, val_data, test_scenarios, device
        )
        # Artifact capture pass
        gat_result = run_gat_inference(
            gat_model, val_data, device, capture_embeddings=True, capture_attention=True
        )
        if not fusion_needs_models:
            del gat_model
            gat_model = None
            cleanup()

    if Path(cfg.checkpoints["vgae"]).exists():
        vgae_model = load_model(cfg, "vgae", vgae_stage, num_ids, in_ch, device)
        all_metrics["vgae"], test_metrics["vgae"], best_thresh = _evaluate_vgae(
            vgae_model, val_data, test_scenarios, device
        )
        # Artifact capture pass
        vgae_result = run_vgae_inference(
            vgae_model, val_data, device, capture_embeddings=True, capture_components=True
        )
        if not fusion_needs_models:
            del vgae_model
            vgae_model = None
            cleanup()

    if fusion_needs_models and vgae_model is not None and gat_model is not None:
        all_metrics["fusion"], test_metrics["fusion"], fusion_result = _evaluate_fusion(
            cfg, gat_model, vgae_model, val_data, test_scenarios, device
        )
        del vgae_model, gat_model
        cleanup()

    if cfg.temporal.enabled and Path(cfg.checkpoints["temporal"]).exists():
        temporal_m = _evaluate_temporal(cfg, val_data, num_ids, in_ch, device, gat_stage)
        if temporal_m is not None:
            all_metrics["temporal"] = temporal_m

    # ---- CKA (KD runs only) ----
    if cfg.has_kd:
        try:
            from .cka import compute_and_save_cka
            compute_and_save_cka(cfg, val_data, device, num_ids, in_ch, Path.cwd())
        except Exception as e:
            log.warning("cka_computation_failed", error=str(e))

    # ---- Persist artifacts ----
    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    from .callbacks import EvalArtifactCallback
    cb = EvalArtifactCallback()
    cb.gat_result = gat_result
    cb.vgae_result = vgae_result
    cb.fusion_result = fusion_result
    cb._save_embeddings(Path.cwd())
    cb._save_attention(Path.cwd())
    cb._save_dqn_policy(Path.cwd())

    cleanup()
    return {"metrics": all_metrics}


# ---------------------------------------------------------------------------
# Per-model evaluators
# ---------------------------------------------------------------------------


def _log_core_metrics(name: str, metrics: dict) -> None:
    log.info(
        "val_metrics",
        model=name,
        **{k: round(v, 4) for k, v in metrics["core"].items() if isinstance(v, float)},
    )


def _evaluate_gat(gat, val_data, test_scenarios, device) -> tuple[dict, dict]:
    """Evaluate GAT on validation + test scenarios. Returns (val_metrics, test_metrics)."""
    result = run_gat_inference(gat, val_data, device)
    val_m = compute_metrics(result.labels, result.preds, result.scores)
    _log_core_metrics("GAT", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            tr = run_gat_inference(gat, tdata, device)
            test_m[scenario] = compute_metrics(tr.labels, tr.preds, tr.scores)
            log.info(
                "gat_test_result",
                scenario=scenario,
                accuracy=round(test_m[scenario]["core"]["accuracy"], 4),
                f1=round(test_m[scenario]["core"]["f1"], 4),
            )
    return val_m, test_m


def _evaluate_vgae(vgae, val_data, test_scenarios, device) -> tuple[dict, dict, float]:
    """Evaluate VGAE on validation + test. Returns (val_metrics, test_metrics, threshold)."""
    result = run_vgae_inference(vgae, val_data, device)
    best_thresh, youden_j, vgae_preds = _vgae_threshold(result.labels, result.errors)
    val_m = compute_metrics(result.labels, vgae_preds, result.errors)
    val_m["core"]["optimal_threshold"] = best_thresh
    val_m["core"]["youden_j"] = youden_j
    _log_core_metrics("VGAE", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            tr = run_vgae_inference(vgae, tdata, device)
            tp = (tr.errors > best_thresh).astype(int)
            test_m[scenario] = compute_metrics(tr.labels, tp, tr.errors)
            test_m[scenario]["core"]["threshold_from_val"] = best_thresh
            log.info(
                "vgae_test_result",
                scenario=scenario,
                accuracy=round(test_m[scenario]["core"]["accuracy"], 4),
                f1=round(test_m[scenario]["core"]["f1"], 4),
            )
    return val_m, test_m, best_thresh


def _evaluate_fusion(
    cfg, gat, vgae, val_data, test_scenarios, device
) -> tuple[dict, dict, "FusionResult"]:
    """Evaluate DQN/MLP/WeightedAvg fusion. Returns (val_metrics, test_metrics, result)."""
    from .eval_types import FusionResult

    models = {"vgae": vgae, "gat": gat}
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)

    from graphids.core.models.dqn import EnhancedDQNFusionAgent

    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    agent = EnhancedDQNFusionAgent.from_config(fusion_cfg, device=str(device), inference=True)
    agent.load_checkpoint(torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True))

    result = run_fusion_inference(agent, val_cache)
    val_m = compute_metrics(result.labels, result.preds, result.scores)
    _log_core_metrics("Fusion", val_m)

    test_m = {}
    if test_scenarios:
        for scenario, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
            tr = run_fusion_inference(agent, tc)
            test_m[scenario] = compute_metrics(tr.labels, tr.preds, tr.scores)
            log.info(
                "fusion_test_result",
                scenario=scenario,
                accuracy=round(test_m[scenario]["core"]["accuracy"], 4),
                f1=round(test_m[scenario]["core"]["f1"], 4),
            )
    return val_m, test_m, result


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
        temporal_model.load_state_dict(torch.load(
            cfg.checkpoints["temporal"], map_location="cpu", weights_only=True,
        ))
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

        metrics = compute_metrics(np.array(t_labels), np.array(t_preds))
        _log_core_metrics("Temporal", metrics)

        del temporal_model, gat_for_temporal
        cleanup()
        return metrics
    except Exception as e:
        log.warning("temporal_evaluation_failed", error=str(e))
        return None


def probe_embedding_dim(model, sample_graph, device) -> int:
    """Probe a model's embedding dimension using a single forward pass."""
    with torch.no_grad():
        probe = sample_graph.clone().to(device)
        _, emb = model(probe, return_embedding=True)
        return emb.shape[-1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _load_test_data(cfg: PipelineConfig) -> dict:
    """Load held-out test graphs per scenario (cached)."""
    from graphids.core.preprocessing import PreprocessingPipeline

    return PreprocessingPipeline(cfg).load_test_scenarios()


def _vgae_threshold(labels, errors):
    """Find optimal anomaly-detection threshold via Youden's J statistic."""
    from torchmetrics.functional.classification import binary_roc

    fpr_v, tpr_v, thresholds_v = binary_roc(
        torch.as_tensor(errors, dtype=torch.float),
        torch.as_tensor(labels, dtype=torch.long),
    )
    j_scores = tpr_v - fpr_v
    best_idx = torch.argmax(j_scores).item()
    best_thresh = (
        float(thresholds_v[best_idx]) if best_idx < len(thresholds_v) else float(np.median(errors))
    )
    preds = (errors > best_thresh).astype(int)
    return best_thresh, float(j_scores[best_idx]), preds


def compute_metrics(labels, preds, scores=None) -> dict:
    """Compute classification metrics using torchmetrics MetricCollection.

    Core metrics via MetricCollection (GPU-native, no sklearn).
    Custom: detection-at-FPR thresholds (no torchmetrics equivalent).
    """
    from torchmetrics import MetricCollection
    from torchmetrics.classification import (
        BinaryAccuracy,
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

    mc = MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "precision": BinaryPrecision(),
            "recall": BinaryRecall(),
            "f1": BinaryF1Score(),
            "specificity": BinarySpecificity(),
            "mcc": BinaryMatthewsCorrCoef(),
            "cm": BinaryConfusionMatrix(),
        }
    )
    r = mc(preds_t, labels_t)
    cm = r.pop("cm")
    tn, fp, fn, tp = cm.ravel().tolist()

    core = {k: v.item() for k, v in r.items()}
    core["balanced_accuracy"] = (core["recall"] + core["specificity"]) / 2
    core["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    core["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    core["n_samples"] = int(len(labels))
    core["confusion_matrix"] = cm.tolist()

    additional: dict = {
        "kappa": BinaryCohenKappa()(preds_t, labels_t).item(),
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
