"""Evaluation stage: runs inference on validation and test data using Lightning test loop."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch

from .callbacks import save_attention, save_dqn_policy, save_embeddings
from .data_loading import cache_predictions, cleanup, load_data
from .eval_inference import (
    capture_gat_artifacts,
    capture_vgae_artifacts,
    find_vgae_threshold,
    run_fusion_inference,
    test_model,
)
from .modules import GATModule, VGAEModule
from .trainer_factory import load_frozen_cfg, load_model

log = structlog.get_logger()


def evaluate(cfg) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Uses Lightning's trainer.test() with torchmetrics for metric computation.
    Returns {"metrics": {...}} with per-model and per-test-scenario results.
    """
    pl.seed_everything(cfg.seed)
    train_data, val_data, num_ids, in_ch = load_data(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    from graphids.core.preprocessing import PreprocessingPipeline
    test_scenarios = PreprocessingPipeline(cfg).load_test_scenarios()

    all_metrics: dict = {}
    test_metrics: dict = {}
    gat_result, vgae_result, fusion_result = None, None, None

    fusion_needs_models = (
        Path(cfg.checkpoints["dqn"]).exists()
        and Path(cfg.checkpoints["vgae"]).exists()
        and Path(cfg.checkpoints["gat"]).exists()
    )

    # ---- GAT evaluation ----
    gat_model = None
    if Path(cfg.checkpoints["gat"]).exists():
        gat_model = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)
        gat_module = GATModule(cfg, num_ids, in_ch)
        gat_module.model = gat_model  # use loaded weights

        all_metrics["gat"] = test_model(gat_module, val_data)
        _log_metrics("GAT", all_metrics["gat"])

        if test_scenarios:
            test_metrics["gat"] = {}
            for name, tdata in test_scenarios.items():
                gat_module.test_metrics.reset()
                test_metrics["gat"][name] = test_model(gat_module, tdata)

        # Artifact capture (separate pass)
        gat_result = capture_gat_artifacts(gat_model, val_data, device)

        if not fusion_needs_models:
            del gat_model
            gat_model = None
            cleanup()

    # ---- VGAE evaluation ----
    vgae_model = None
    if Path(cfg.checkpoints["vgae"]).exists():
        vgae_model = load_model(cfg, "vgae", "autoencoder", num_ids, in_ch, device)
        vgae_module = VGAEModule(cfg, num_ids, in_ch)
        vgae_module.model = vgae_model  # use loaded weights

        # Find threshold on val, then evaluate with it
        threshold, youden_j = find_vgae_threshold(vgae_module, val_data)
        vgae_module.test_threshold = threshold
        vgae_module._test_errors.clear()
        vgae_module._test_labels.clear()
        vgae_module.test_metrics.reset()
        all_metrics["vgae"] = test_model(vgae_module, val_data)
        all_metrics["vgae"]["core"]["optimal_threshold"] = threshold
        all_metrics["vgae"]["core"]["youden_j"] = youden_j
        _log_metrics("VGAE", all_metrics["vgae"])

        if test_scenarios:
            test_metrics["vgae"] = {}
            for name, tdata in test_scenarios.items():
                vgae_module.test_metrics.reset()
                vgae_module._test_errors.clear()
                vgae_module._test_labels.clear()
                test_metrics["vgae"][name] = test_model(vgae_module, tdata)

        # Artifact capture
        vgae_result = capture_vgae_artifacts(vgae_model, val_data, device)

        if not fusion_needs_models:
            del vgae_model
            vgae_model = None
            cleanup()

    # ---- Fusion evaluation ----
    if fusion_needs_models and vgae_model is not None and gat_model is not None:
        fusion_val, fusion_test, fusion_result = _evaluate_fusion(
            cfg, gat_model, vgae_model, val_data, test_scenarios, device,
        )
        all_metrics["fusion"] = fusion_val
        if fusion_test:
            test_metrics["fusion"] = fusion_test
        del vgae_model, gat_model
        cleanup()

    # ---- Temporal evaluation ----
    if cfg.temporal.enabled and Path(cfg.checkpoints["temporal"]).exists():
        temporal_m = _evaluate_temporal(cfg, val_data, num_ids, in_ch, device)
        if temporal_m is not None:
            all_metrics["temporal"] = temporal_m

    # ---- CKA (KD runs only) ----
    if any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        try:
            from .cka import compute_and_save_cka
            compute_and_save_cka(cfg, val_data, device, num_ids, in_ch, Path.cwd())
        except Exception as e:
            log.warning("cka_failed", error=str(e))

    # ---- Persist artifacts ----
    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    save_embeddings(Path.cwd(), vgae_result, gat_result)
    save_attention(Path.cwd(), gat_result)
    save_dqn_policy(Path.cwd(), fusion_result)

    cleanup()
    return {"metrics": all_metrics}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_metrics(name: str, metrics: dict) -> None:
    log.info("val_metrics", model=name,
             **{k: round(v, 4) for k, v in metrics["core"].items() if isinstance(v, float)})


def _evaluate_fusion(cfg, gat, vgae, val_data, test_scenarios, device):
    """Evaluate DQN fusion. Returns (val_metrics, test_metrics, FusionResult)."""
    from .eval_inference import run_fusion_inference

    models = {"vgae": vgae, "gat": gat}
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples)

    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    method = fusion_cfg.fusion.method if hasattr(fusion_cfg, "fusion") else cfg.fusion.method

    if method == "bandit":
        from graphids.core.models.bandit import NeuralLinUCBAgent
        agent = NeuralLinUCBAgent.from_config(fusion_cfg, device=str(device))
    else:
        from graphids.core.models.dqn import EnhancedDQNFusionAgent
        agent = EnhancedDQNFusionAgent.from_config(fusion_cfg, device=str(device), inference=True)

    agent.load_checkpoint(torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True))

    result = run_fusion_inference(agent, val_cache)

    # Manual metrics for fusion (DQN agent isn't a Lightning module)
    from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
    labels_t = torch.as_tensor(result.labels, dtype=torch.long)
    preds_t = torch.as_tensor(result.preds, dtype=torch.long)
    scores_t = torch.as_tensor(result.scores, dtype=torch.float)
    val_m = {
        "core": {
            "accuracy": BinaryAccuracy()(preds_t, labels_t).item(),
            "f1": BinaryF1Score()(preds_t, labels_t).item(),
            "auc": BinaryAUROC()(scores_t, labels_t).item(),
        },
        "additional": {},
    }
    _log_metrics("Fusion", val_m)

    test_m = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples)
            tr = run_fusion_inference(agent, tc)
            lt = torch.as_tensor(tr.labels, dtype=torch.long)
            pt = torch.as_tensor(tr.preds, dtype=torch.long)
            st = torch.as_tensor(tr.scores, dtype=torch.float)
            test_m[name] = {
                "core": {
                    "accuracy": BinaryAccuracy()(pt, lt).item(),
                    "f1": BinaryF1Score()(pt, lt).item(),
                    "auc": BinaryAUROC()(st, lt).item(),
                },
                "additional": {},
            }
    return val_m, test_m, result


def _evaluate_temporal(cfg, val_data, num_ids, in_ch, device) -> dict | None:
    """Evaluate temporal model. Returns metrics dict or None on failure."""
    try:
        from graphids.core.models.temporal import TemporalGraphClassifier
        from graphids.core.preprocessing._temporal import TemporalGrouper

        gat = load_model(cfg, "gat", "curriculum", num_ids, in_ch, device)
        from .eval_inference import GATTestModule
        # Probe embedding dim
        with torch.no_grad():
            probe = val_data[0].clone().to(device)
            _, emb = gat(probe, return_embedding=True)
            spatial_dim = emb.shape[-1]

        tc = cfg.temporal
        temporal_model = TemporalGraphClassifier(
            spatial_encoder=gat, spatial_dim=spatial_dim,
            temporal_hidden=tc.temporal_hidden, temporal_heads=tc.temporal_heads,
            temporal_layers=tc.temporal_layers, max_seq_len=tc.temporal_window,
            freeze_spatial=True, num_classes=2,
        ).to(device)
        temporal_model.load_state_dict(torch.load(
            cfg.checkpoints["temporal"], map_location="cpu", weights_only=True,
        ))
        temporal_model.eval()

        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        val_sequences = grouper.group(val_data)
        if not val_sequences:
            return None

        from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
        import numpy as np
        t_preds, t_labels = [], []
        with torch.no_grad():
            for seq_obj in val_sequences:
                moved = [g.clone().to(device) for g in seq_obj.graphs]
                logits = temporal_model([[g for g in moved]])
                t_preds.append(logits.argmax(dim=1)[0].item())
                t_labels.append(seq_obj.y)

        lt = torch.tensor(t_labels)
        pt = torch.tensor(t_preds)
        metrics = {
            "core": {
                "accuracy": BinaryAccuracy()(pt, lt).item(),
                "f1": BinaryF1Score()(pt, lt).item(),
            },
            "additional": {},
        }
        _log_metrics("Temporal", metrics)
        del temporal_model, gat
        cleanup()
        return metrics
    except Exception as e:
        log.warning("temporal_eval_failed", error=str(e))
        return None
