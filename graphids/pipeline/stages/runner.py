"""Stage runner: dispatch to LightningCLI (training) or multi-model eval loop.

Training lifecycle (run dirs, config persistence, DuckDB catalog) handled by
Lightning callbacks in graphids.pipeline.callbacks. Trainer construction via
GraphIDSCLI in graphids.pipeline.cli.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
from torch.utils.data import DataLoader, TensorDataset

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage(cfg, stage: str) -> dict:
    """Dispatch to train or evaluate."""
    from graphids.config import STAGES, to_namespace

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    cfg = to_namespace(cfg)
    if stage == "evaluation":
        return _evaluate(cfg)
    return _train(cfg, stage)


def load_model(cfg, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model's inner nn.Module. Used by eval, fusion, curriculum."""
    from graphids.core.models._training import load_inner_model
    model, _ = load_inner_model(model_type, Path(cfg.checkpoints[model_type]), device)
    return model


# ---------------------------------------------------------------------------
# Stage → class dispatch tables
# ---------------------------------------------------------------------------


def _build_dm(cfg, stage: str):
    """Build the right DataModule for a stage. Returns (dm, device)."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    if stage == "curriculum":
        from graphids.core.preprocessing.curriculum import CurriculumDataModule
        dm = CurriculumDataModule.from_cfg(cfg)
        return dm, device
    if stage == "fusion":
        from graphids.core.preprocessing import FusionDataModule
        dm = FusionDataModule(cfg, load_model_fn=load_model)
        dm.setup("fit")
        return dm, dm.device
    if stage == "temporal":
        from graphids.core.preprocessing import TemporalDataModule
        dm = TemporalDataModule(cfg, load_model_fn=load_model)
        dm.setup("fit")
        return dm, dm.device

    from graphids.core.preprocessing import CANBusDataModule
    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup("fit")
    dm.populate_config(cfg)
    return dm, device


def _build_module(cfg, stage: str, device, dm=None):
    """Build the right LightningModule for a stage."""
    if stage == "fusion":
        from graphids.core.models.fusion_baselines import build_fusion_module
        return build_fusion_module(cfg, device)
    if stage == "temporal":
        from graphids.core.models.temporal import TemporalLightningModule
        return TemporalLightningModule.from_datamodule(cfg, dm)

    from graphids.core.models.registry import get_module_cls

    module_cls = get_module_cls(cfg.model_type)
    return module_cls(cfg)


# ---------------------------------------------------------------------------
# Training — pl.Trainer.fit() with callback-driven lifecycle
# ---------------------------------------------------------------------------


def _train(cfg, stage: str) -> dict:
    """Seed, build DM + module, train via LightningCLI."""
    if stage == "temporal" and not cfg.temporal.enabled:
        log.warning("temporal.enabled=False, skipping")
        return {"status": "skipped", "reason": "temporal.enabled=False"}

    from graphids.pipeline.cli import train_stage
    return train_stage(cfg, stage)


# ---------------------------------------------------------------------------
# Evaluation — pl.Trainer.test() in a multi-model loop
# ---------------------------------------------------------------------------

_EVAL_ORDER = ["gat", "vgae", "dgi", "fusion", "temporal"]


def _evaluate(cfg) -> dict:
    """Evaluate trained models: Trainer.test() per model + artifact generation."""
    from graphids.config import compute_identity_hash

    from graphids.core.models._training import eval_with_scenarios, gpu_cleanup, test_model
    from graphids.core.models.fusion_baselines import run_fusion_inference
    from graphids.core.models.registry import get_module_cls
    from graphids.core.preprocessing import CANBusDataModule

    pl.seed_everything(cfg.seed, workers=True)

    # Set up eval run directory
    identity = compute_identity_hash("evaluation", cfg)
    run_dir = (
        Path(cfg._output_base)
        / f"{cfg.model_type}_{cfg.scale}_evaluation{identity}"
        / f"seed_{cfg.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage="evaluation", seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        run_dir=str(run_dir),
    )

    dm = CANBusDataModule.from_cfg(cfg)
    dm.setup()
    dm.populate_config(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    val_data = list(dm.val_dataset)
    test_scenarios = {name: list(ds) for name, ds in dm.test_datasets.items()} or None

    all_metrics: dict = {}
    test_metrics: dict = {}
    artifacts: dict = {}
    bs = cfg.evaluation.batch_size

    for model_name in _EVAL_ORDER:
        ckpt_key = "dqn" if model_name == "fusion" else model_name
        if model_name == "temporal" and not cfg.temporal.enabled:
            continue
        ckpt = cfg.checkpoints.get(ckpt_key)
        if not ckpt or not Path(ckpt).exists():
            continue

        if model_name == "fusion":
            if not all(Path(cfg.checkpoints.get(k, "")).exists() for k in ("dqn", "vgae", "gat")):
                continue
            result = _eval_fusion(cfg, val_data, test_scenarios, device, bs)
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

        log.info("val_metrics", model=model_name.upper(),
                 **{k: round(v, 4) for k, v in result["val_metrics"].items() if isinstance(v, float)})
        all_metrics[model_name] = result["val_metrics"]
        if result.get("test_metrics"):
            test_metrics[model_name] = result["test_metrics"]
        if result.get("artifacts"):
            artifacts[model_name] = result["artifacts"]

    # Artifact generation (embeddings, attention, CKA, loss landscape)
    from graphids.core.artifacts import generate_all
    generate_all(cfg, val_data, device, Path.cwd(), artifacts, load_model_fn=load_model)

    if test_metrics:
        all_metrics["test"] = {k: v for k, v in test_metrics.items() if v}

    Path("metrics.json").write_text(json.dumps(all_metrics, indent=2, default=float))
    log.info("metrics_saved", path=str(Path.cwd() / "metrics.json"))

    gpu_cleanup()
    return {"metrics": all_metrics}


def _eval_fusion(cfg, val_data, test_scenarios, device, bs) -> dict:
    """Fusion eval: load VGAE+GAT, cache predictions, Trainer.test() on fusion module."""
    from graphids.core.models._training import gpu_cleanup, test_model
    from graphids.core.models.fusion_baselines import MLPFusionModule, RLFusionModule, WeightedAvgModule, run_fusion_inference
    from graphids.core.preprocessing import FusionDataModule

    vgae = load_model(cfg, "vgae", device)
    gat = load_model(cfg, "gat", device)
    models = {"vgae": vgae, "gat": gat}

    val_cache = FusionDataModule.cache_predictions(models, val_data, device, cfg.fusion.max_val_samples, batch_size=bs)

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
