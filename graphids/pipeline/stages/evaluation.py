"""Evaluation stage: runs inference on validation and test data using Lightning test loop.

All models evaluate via trainer.test(module, dataloaders=...) — one pattern.
Dispatcher iterates EVAL_ORDER and calls per-model eval functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.core.preprocessing import CANBusDataModule

import gc

from .fusion import cache_predictions
from .eval_inference import (
    FusionResult,
    GATResult,
    VGAEResult,
    capture_gat_artifacts,
    capture_vgae_artifacts,
    find_vgae_threshold,
    run_fusion_inference,
    test_model,
)
from graphids.core.models.dgi import DGIModule
from graphids.core.models.gat import GATModule
from graphids.core.models.vgae import VGAEModule
from .trainer_factory import load_frozen_cfg, load_model

log = structlog.get_logger()

EVAL_ORDER = ["gat", "vgae", "dgi", "fusion", "temporal"]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def evaluate(cfg) -> dict:
    """Evaluate trained model(s) on validation and held-out test data.

    Uses Lightning's trainer.test() with torchmetrics for metric computation.
    Returns {"metrics": {...}} with per-model and per-test-scenario results.
    """
    pl.seed_everything(cfg.seed)
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup()  # both fit + test
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}

    eval_fns = {
        "gat": eval_gat,
        "vgae": lambda cfg, val, ts, dev: _eval_unsupervised(
            cfg, val, ts, dev, "vgae", VGAEModule, capture_fn=capture_vgae_artifacts,
        ),
        "dgi": lambda cfg, val, ts, dev: _eval_unsupervised(
            cfg, val, ts, dev, "dgi", DGIModule,
        ),
        "fusion": eval_fusion,
        "temporal": eval_temporal,
    }

    for model_name in EVAL_ORDER:
        ckpt_key = "dqn" if model_name == "fusion" else model_name
        if model_name == "temporal" and not cfg.temporal.enabled:
            continue
        ckpt = cfg.checkpoints.get(ckpt_key)
        if not ckpt or not Path(ckpt).exists():
            continue
        # Fusion needs all 3 checkpoints
        if model_name == "fusion" and not _fusion_checkpoints_exist(cfg):
            continue

        result = eval_fns[model_name](cfg, val_data, test_scenarios, device)
        if result is None:
            continue

        all_metrics[model_name] = result["val_metrics"]
        if result.get("test_metrics"):
            test_metrics[model_name] = result["test_metrics"]
        if result.get("artifacts"):
            artifacts[model_name] = result["artifacts"]

    # CKA (KD runs only)
    if any(a.type == "kd" for a in cfg.get("auxiliaries", [])):
        try:
            from .cka import compute_and_save_cka
            compute_and_save_cka(cfg, val_data, device, Path.cwd(), max_samples=cfg.evaluation.cka_max_samples)
        except Exception as e:
            log.warning("cka_failed", error=str(e))

    # Persist artifacts
    _save_embeddings(
        Path.cwd(),
        artifacts.get("vgae"),
        artifacts.get("gat"),
    )
    _save_attention(Path.cwd(), artifacts.get("gat"))
    _save_dqn_policy(Path.cwd(), artifacts.get("fusion"))

    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    Path("metrics.json").write_text(json.dumps(all_metrics, indent=2, default=float))
    log.info("metrics_saved", path=str(Path.cwd() / "metrics.json"))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"metrics": all_metrics}


def _fusion_checkpoints_exist(cfg) -> bool:
    return all(
        Path(cfg.checkpoints.get(k, "")).exists()
        for k in ("dqn", "vgae", "gat")
    )


# ---------------------------------------------------------------------------
# Per-model eval functions
# ---------------------------------------------------------------------------


def eval_gat(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate GAT via trainer.test() + capture artifacts."""
    gat_model = load_model(cfg, "gat", cfg.gat_stage, device)
    module = GATModule(cfg)
    module.model = gat_model

    bs = cfg.evaluation.batch_size
    val_metrics = test_model(module, val_data, batch_size=bs)
    _log_metrics("GAT", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tdata, batch_size=bs)

    gat_result = capture_gat_artifacts(gat_model, val_data, device, batch_size=bs, attention_limit=cfg.evaluation.attention_sample_limit)

    del gat_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": gat_result,
    }


def _eval_unsupervised(cfg, val_data, test_scenarios, device, model_name, module_cls, capture_fn=None) -> dict:
    """Evaluate an unsupervised model (VGAE or DGI): threshold search on val, then test."""
    model = load_model(cfg, model_name, "autoencoder", device)
    module = module_cls(cfg)
    module.model = model

    bs = cfg.evaluation.batch_size
    threshold, youden_j = find_vgae_threshold(module, val_data, batch_size=bs)
    module.test_threshold = threshold

    # Clear accumulation lists — attribute names differ by module
    for attr in ("_test_errors", "_test_scores", "_test_labels"):
        acc = getattr(module, attr, None)
        if acc is not None:
            acc.clear()
    module.test_metrics.reset()

    val_metrics = test_model(module, val_data, batch_size=bs)
    val_metrics["optimal_threshold"] = threshold
    val_metrics["youden_j"] = youden_j
    _log_metrics(model_name.upper(), val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            for attr in ("_test_errors", "_test_scores", "_test_labels"):
                acc = getattr(module, attr, None)
                if acc is not None:
                    acc.clear()
            scenario_metrics[name] = test_model(module, tdata, batch_size=bs)

    artifacts = capture_fn(model, val_data, device, batch_size=bs) if capture_fn else None

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": artifacts,
    }


def eval_fusion(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate fusion agent via Lightning test loop."""
    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", cfg.gat_stage, device)
    models = {"vgae": vgae, "gat": gat}

    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples, batch_size=cfg.evaluation.batch_size)
    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    method = fusion_cfg.fusion.method if hasattr(fusion_cfg, "fusion") else cfg.fusion.method

    # Load trained fusion module via unified loader
    from graphids.core.models.fusion_baselines import load_fusion_module
    ckpt = torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True)
    module = load_fusion_module(ckpt, fusion_cfg, device=str(device))

    # Eval via trainer.test()
    val_loader = DataLoader(
        TensorDataset(val_cache["states"], val_cache["labels"]),
        batch_size=cfg.evaluation.batch_size, shuffle=False,
    )
    val_metrics = test_model(module, val_loader)
    _log_metrics("Fusion", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples, batch_size=cfg.evaluation.batch_size)
            tl = DataLoader(
                TensorDataset(tc["states"], tc["labels"]),
                batch_size=cfg.evaluation.batch_size, shuffle=False,
            )
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tl)

    # Artifact capture (q-values / policy JSON) — only for DQN/bandit
    fusion_result = None
    if method in ("dqn", "bandit"):
        fusion_result = run_fusion_inference(module.agent, val_cache)

    del vgae, gat
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "val_metrics": val_metrics,
        "test_metrics": scenario_metrics,
        "artifacts": fusion_result,
    }


def eval_temporal(cfg, val_data, test_scenarios, device) -> dict | None:
    """Evaluate temporal model via Lightning test loop."""
    try:
        from graphids.core.models.temporal import TemporalGraphClassifier, TemporalLightningModule
        from graphids.core.preprocessing._temporal import (
            TemporalGraphDataset,
            TemporalGrouper,
            collate_temporal,
        )

        gat = load_model(cfg, "gat", cfg.gat_stage, device)

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
            freeze_spatial=True, num_classes=cfg.num_classes,
        ).to(device)
        temporal_model.load_state_dict(torch.load(
            cfg.checkpoints["temporal"], map_location="cpu", weights_only=True,
        ))

        module = TemporalLightningModule(temporal_model, cfg)

        grouper = TemporalGrouper(window=tc.temporal_window, stride=tc.temporal_stride)
        val_sequences = grouper.group(val_data)
        if not val_sequences:
            return None

        val_ds = TemporalGraphDataset(val_sequences, device)
        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
        val_metrics = test_model(module, val_loader)
        _log_metrics("Temporal", val_metrics)

        del temporal_model, gat
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"val_metrics": val_metrics, "test_metrics": {}, "artifacts": None}
    except Exception:
        log.exception("temporal_eval_failed")
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_metrics(name: str, metrics: dict) -> None:
    log.info(
        "val_metrics", model=name,
        **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
    )


def _save_embeddings(
    out: Path, vgae_result: VGAEResult | None, gat_result: GATResult | None,
) -> None:
    embed_data: dict[str, np.ndarray] = {}
    if vgae_result is not None:
        if vgae_result.embeddings is not None:
            embed_data["vgae_z"] = vgae_result.embeddings
            embed_data["vgae_labels"] = vgae_result.labels
            embed_data["vgae_errors"] = vgae_result.errors
            embed_data["vgae_attack_types"] = vgae_result.attack_types
        if vgae_result.components is not None:
            for name, arr in vgae_result.components.items():
                embed_data[f"vgae_error_{name}"] = arr
    if gat_result is not None and gat_result.embeddings is not None:
        embed_data["gat_emb"] = gat_result.embeddings
        embed_data["gat_labels"] = gat_result.labels
        embed_data["gat_attack_types"] = gat_result.attack_types
    if embed_data:
        path = out / "embeddings.npz"
        np.savez_compressed(path, **embed_data)
        log.info("embeddings_saved", path=str(path))


def _save_attention(out: Path, gat_result: GATResult | None) -> None:
    if gat_result is None or not gat_result.attention:
        return
    attn_export: dict = {}
    for i, entry in enumerate(gat_result.attention):
        prefix = f"sample_{i}"
        attn_export[f"{prefix}_graph_idx"] = entry["graph_idx"]
        attn_export[f"{prefix}_label"] = entry["label"]
        attn_export[f"{prefix}_edge_index"] = entry["edge_index"]
        attn_export[f"{prefix}_node_features"] = entry["node_features"]
        for layer_idx, aw in enumerate(entry["attention_weights"]):
            attn_export[f"{prefix}_layer_{layer_idx}_alpha"] = aw
    attn_export["n_samples"] = len(gat_result.attention)
    path = out / "attention_weights.npz"
    np.savez_compressed(path, **attn_export)
    log.info("attention_weights_saved", samples=len(gat_result.attention), path=str(path))


def _save_dqn_policy(out: Path, fusion_result: FusionResult | None) -> None:
    if fusion_result is None:
        return
    alphas = fusion_result.scores.tolist()
    labels = fusion_result.labels.tolist()
    alpha_by_label: dict[str, list] = {"normal": [], "attack": []}
    for a, lbl in zip(alphas, labels):
        alpha_by_label["normal" if lbl == 0 else "attack"].append(a)
    policy_data = {
        "alphas": alphas, "labels": labels,
        "alpha_by_label": alpha_by_label,
        "q_values": fusion_result.q_values.tolist(),
    }
    path = out / "dqn_policy.json"
    path.write_text(json.dumps(policy_data, indent=2))
    log.info("dqn_policy_saved", path=str(path))
