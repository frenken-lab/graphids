"""Evaluation stage: runs inference on validation and test data.

Dispatcher iterates EVAL_ORDER, delegates to module classmethods for standard
models, and handles fusion (composite, multi-model) inline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

from graphids.core.models._training import gpu_cleanup, test_model
from graphids.core.models.fusion_baselines import run_fusion_inference
from graphids.core.models.registry import get_module_cls
from graphids.core.preprocessing import CANBusDataModule, FusionDataModule

from .trainer_factory import load_model

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
    dm.setup()
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}

    for model_name in EVAL_ORDER:
        ckpt_key = "dqn" if model_name == "fusion" else model_name
        if model_name == "temporal" and not cfg.temporal.enabled:
            continue
        ckpt = cfg.checkpoints.get(ckpt_key)
        if not ckpt or not Path(ckpt).exists():
            continue
        if model_name == "fusion" and not _fusion_checkpoints_exist(cfg):
            continue

        if model_name == "fusion":
            result = _eval_fusion(cfg, val_data, test_scenarios, device)
        elif model_name == "temporal":
            from graphids.core.models.temporal import TemporalLightningModule
            result = TemporalLightningModule.evaluate(
                cfg, val_data, test_scenarios, device, load_model_fn=load_model,
            )
        else:
            module_cls = get_module_cls(model_name)
            result = module_cls.evaluate(
                cfg, val_data, test_scenarios, device, load_model_fn=load_model,
            )

        if result is None:
            continue

        _log_metrics(model_name.upper(), result["val_metrics"])
        all_metrics[model_name] = result["val_metrics"]
        if result.get("test_metrics"):
            test_metrics[model_name] = result["test_metrics"]
        if result.get("artifacts"):
            artifacts[model_name] = result["artifacts"]

    # Generate all derived artifacts (embeddings, attention, CKA, loss landscape, etc.)
    from graphids.core.artifacts import generate_all
    generate_all(cfg, val_data, device, Path.cwd(), artifacts, load_model_fn=load_model)

    test_metrics = {k: v for k, v in test_metrics.items() if v}
    if test_metrics:
        all_metrics["test"] = test_metrics

    Path("metrics.json").write_text(json.dumps(all_metrics, indent=2, default=float))
    log.info("metrics_saved", path=str(Path.cwd() / "metrics.json"))

    gpu_cleanup()
    return {"metrics": all_metrics}


def _fusion_checkpoints_exist(cfg) -> bool:
    return all(Path(cfg.checkpoints.get(k, "")).exists() for k in ("dqn", "vgae", "gat"))


# ---------------------------------------------------------------------------
# Fusion eval (composite — loads multiple models, stays in pipeline)
# ---------------------------------------------------------------------------


def _eval_fusion(cfg, val_data, test_scenarios, device) -> dict:
    """Evaluate fusion agent via Lightning test loop."""
    vgae = load_model(cfg, "vgae", device)
    gat = load_model(cfg, "gat", device)
    models = {"vgae": vgae, "gat": gat}

    bs = cfg.evaluation.batch_size
    val_cache = FusionDataModule.cache_predictions(models, val_data, device, cfg.fusion.max_val_samples, batch_size=bs)

    # Load fusion module via Lightning — method dispatch is handled by the checkpoint's hparams
    from graphids.core.models.fusion_baselines import MLPFusionModule, RLFusionModule, WeightedAvgModule
    ckpt_path = cfg.checkpoints["dqn"]
    method = cfg.fusion.method
    _fusion_cls = {"mlp": MLPFusionModule, "weighted_avg": WeightedAvgModule}.get(method, RLFusionModule)
    module = _fusion_cls.load_from_checkpoint(ckpt_path, map_location=str(device), weights_only=True)

    val_loader = DataLoader(
        TensorDataset(val_cache["states"], val_cache["labels"]),
        batch_size=bs, shuffle=False,
    )
    val_metrics = test_model(module, val_loader)

    scenario_metrics = {}
    if test_scenarios:
        for name, tdata in test_scenarios.items():
            tc = FusionDataModule.cache_predictions(models, tdata, device, cfg.fusion.max_val_samples, batch_size=bs)
            tl = DataLoader(TensorDataset(tc["states"], tc["labels"]), batch_size=bs, shuffle=False)
            module.test_metrics.reset()
            scenario_metrics[name] = test_model(module, tl)

    fusion_result = None
    if method in ("dqn", "bandit"):
        fusion_result = run_fusion_inference(module.agent, val_cache)

    gpu_cleanup(vgae, gat)
    return {"val_metrics": val_metrics, "test_metrics": scenario_metrics, "artifacts": fusion_result}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_metrics(name: str, metrics: dict) -> None:
    log.info(
        "val_metrics", model=name,
        **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
    )
