"""Evaluation stage: runs inference on validation and test data.

All models evaluate via trainer.test(module, dataloaders=...) — one pattern.
Dispatcher iterates EVAL_ORDER and calls per-model eval functions.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from graphids.core.models.dgi import DGIModule
from graphids.core.models.fusion_baselines import load_fusion_module, run_fusion_inference
from graphids.core.models.gat import GATModule
from graphids.core.models.vgae import VGAEModule
from graphids.core.preprocessing import CANBusDataModule

from .fusion import cache_predictions
from .trainer_factory import load_frozen_cfg, load_model

log = structlog.get_logger()

EVAL_ORDER = ["gat", "vgae", "dgi", "fusion", "temporal"]


# ---------------------------------------------------------------------------
# Test runner (inlined from eval_inference.py)
# ---------------------------------------------------------------------------


def _make_test_trainer() -> pl.Trainer:
    return pl.Trainer(
        accelerator="auto", devices="auto",
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
    )


def _test_model(module, data, batch_size: int = 256) -> dict:
    """Run trainer.test() on a module and return metrics.

    Args:
        data: Either a list of PyG Data objects (creates PyGDataLoader) or
              a pre-built DataLoader (used as-is, e.g. for fusion tensor batches).
    """
    trainer = _make_test_trainer()
    if isinstance(data, list):
        loader = PyGDataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        loader = data
    results = trainer.test(module, dataloaders=loader, verbose=False)
    metrics = dict(results[0]) if results else {}
    metrics["balanced_accuracy"] = (metrics.get("recall", 0) + metrics.get("specificity", 0)) / 2
    return metrics


def _eval_with_scenarios(module, val_data, test_scenarios, batch_size: int, reset_fn=None) -> tuple[dict, dict]:
    """Run test on val + each test scenario. Returns (val_metrics, scenario_metrics).

    Args:
        reset_fn: Optional callable invoked before each scenario to reset
                  module state beyond test_metrics (e.g. accumulation lists).
    """
    val_metrics = _test_model(module, val_data, batch_size=batch_size)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            module.test_metrics.reset()
            if reset_fn:
                reset_fn()
            scenario_metrics[name] = _test_model(module, tdata, batch_size=batch_size)

    return val_metrics, scenario_metrics


def _gpu_cleanup(*objs):
    """Delete objects and free GPU memory."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    dm.setup()
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}

    eval_fns = {
        "gat": eval_gat,
        "vgae": lambda cfg, val, ts, dev: _eval_unsupervised(cfg, val, ts, dev, "vgae", VGAEModule, capture=True),
        "dgi": lambda cfg, val, ts, dev: _eval_unsupervised(cfg, val, ts, dev, "dgi", DGIModule),
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

    # Generate all derived artifacts (embeddings, attention, CKA, loss landscape, etc.)
    from graphids.core.artifacts import generate_all
    from .trainer_factory import load_model
    generate_all(cfg, val_data, device, Path.cwd(), artifacts, load_model_fn=load_model)

    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    Path("metrics.json").write_text(json.dumps(all_metrics, indent=2, default=float))
    log.info("metrics_saved", path=str(Path.cwd() / "metrics.json"))

    _gpu_cleanup()
    return {"metrics": all_metrics}


def _fusion_checkpoints_exist(cfg) -> bool:
    return all(Path(cfg.checkpoints.get(k, "")).exists() for k in ("dqn", "vgae", "gat"))


# ---------------------------------------------------------------------------
# Per-model eval functions
# ---------------------------------------------------------------------------


def eval_gat(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate GAT via trainer.test() + capture artifacts."""
    gat_model = load_model(cfg, "gat", cfg.gat_stage, device)
    module = GATModule(cfg)
    module.model = gat_model

    bs = cfg.evaluation.batch_size
    val_metrics, scenario_metrics = _eval_with_scenarios(module, val_data, test_scenarios, bs)
    _log_metrics("GAT", val_metrics)

    gat_result = gat_model.capture_artifacts(
        val_data, device, batch_size=bs,
        attention_limit=cfg.evaluation.attention_sample_limit,
    )
    _gpu_cleanup(gat_model)
    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": gat_result}


def _eval_unsupervised(cfg, val_data, test_scenarios, device, model_name, module_cls, capture=False) -> dict:
    """Evaluate an unsupervised model (VGAE or DGI): threshold search on val, then test."""
    model = load_model(cfg, model_name, "autoencoder", device)
    module = module_cls(cfg)
    module.model = model

    bs = cfg.evaluation.batch_size
    threshold, youden_j = module.find_threshold(val_data, batch_size=bs)
    module.test_threshold = threshold

    def _clear_accumulators():
        for attr in ("_test_errors", "_test_scores", "_test_labels"):
            acc = getattr(module, attr, None)
            if acc is not None:
                acc.clear()

    _clear_accumulators()
    module.test_metrics.reset()

    val_metrics, scenario_metrics = _eval_with_scenarios(
        module, val_data, test_scenarios, bs, reset_fn=_clear_accumulators,
    )
    val_metrics["optimal_threshold"] = threshold
    val_metrics["youden_j"] = youden_j
    _log_metrics(model_name.upper(), val_metrics)

    artifacts = model.capture_artifacts(val_data, device, batch_size=bs) if capture else None
    _gpu_cleanup(model)
    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": artifacts}


def eval_fusion(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate fusion agent via Lightning test loop."""
    vgae = load_model(cfg, "vgae", "autoencoder", device)
    gat = load_model(cfg, "gat", cfg.gat_stage, device)
    models = {"vgae": vgae, "gat": gat}

    bs = cfg.evaluation.batch_size
    val_cache = cache_predictions(models, val_data, device, cfg.fusion.max_val_samples, batch_size=bs)
    fusion_cfg = load_frozen_cfg(cfg, "fusion")
    method = fusion_cfg.fusion.method if hasattr(fusion_cfg, "fusion") else cfg.fusion.method

    ckpt = torch.load(cfg.checkpoints["dqn"], map_location="cpu", weights_only=True)
    module = load_fusion_module(ckpt, fusion_cfg, device=str(device))

    val_loader = DataLoader(
        TensorDataset(val_cache["states"], val_cache["labels"]),
        batch_size=bs, shuffle=False,
    )
    val_metrics = _test_model(module, val_loader)
    _log_metrics("Fusion", val_metrics)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = cache_predictions(models, tdata, device, cfg.fusion.max_val_samples, batch_size=bs)
            tl = DataLoader(TensorDataset(tc["states"], tc["labels"]), batch_size=bs, shuffle=False)
            module.test_metrics.reset()
            scenario_metrics[name] = _test_model(module, tl)

    fusion_result = None
    if method in ("dqn", "bandit"):
        fusion_result = run_fusion_inference(module.agent, val_cache)

    _gpu_cleanup(vgae, gat)
    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": fusion_result}


def eval_temporal(cfg, val_data, test_scenarios, device) -> dict | None:
    """Evaluate temporal model via Lightning test loop."""
    try:
        from graphids.core.models.temporal import TemporalGraphClassifier, TemporalLightningModule
        from graphids.core.preprocessing._temporal import TemporalGraphDataset, TemporalGrouper, collate_temporal

        gat = load_model(cfg, "gat", cfg.gat_stage, device)

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

        val_loader = DataLoader(
            TemporalGraphDataset(val_sequences, device),
            batch_size=32, shuffle=False,
            collate_fn=collate_temporal, num_workers=0,
        )
        val_metrics = _test_model(module, val_loader)
        _log_metrics("Temporal", val_metrics)

        _gpu_cleanup(temporal_model, gat)
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
