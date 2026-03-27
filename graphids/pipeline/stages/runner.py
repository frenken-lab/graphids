"""Stage runner: setup, Lightning train/test, catalog.

Training = pl.Trainer.fit(). Evaluation = pl.Trainer.test() in a loop.
Custom logic: identity-hash paths, structlog context, DuckDB catalog, artifact generation.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytorch_lightning as pl
import structlog
import torch
import yaml
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from torch.utils.data import DataLoader, TensorDataset

log = structlog.get_logger()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage(cfg, stage: str) -> dict:
    """Setup run directory, dispatch to train or evaluate, append to catalog."""
    from graphids.config import STAGES, compute_identity_hash, to_namespace

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    cfg = to_namespace(cfg)

    # --- Run directory (identity-hash aware) ---
    identity = compute_identity_hash(stage, cfg)
    run_dir = (
        Path(cfg._output_base)
        / f"{cfg.model_type}_{cfg.scale}_{stage}{identity}"
        / f"seed_{cfg.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        run_dir=str(run_dir),
    )

    # --- Save config + git SHA ---
    (run_dir / "config.yaml").write_text(
        yaml.dump(cfg.as_dict(), default_flow_style=False, sort_keys=False),
    )
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    (run_dir / "run_metadata.json").write_text(json.dumps({"git_sha": sha}, indent=2))

    # --- Dispatch ---
    if stage == "evaluation":
        result = _evaluate(cfg)
    else:
        result = _train(cfg, stage)

    _append_to_catalog(cfg, stage, result, run_dir)
    return result


def load_model(cfg, model_type: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model's inner nn.Module. Used by eval, fusion, curriculum."""
    from graphids.core.models._training import load_inner_model
    model, _ = load_inner_model(model_type, Path(cfg.checkpoints[model_type]), device)
    return model


# ---------------------------------------------------------------------------
# Training — pl.Trainer.fit() with cfg-driven args
# ---------------------------------------------------------------------------


def _train(cfg, stage: str) -> dict:
    """Seed, build DM + module, Trainer.fit(), return metrics."""
    from graphids.core.models._training import gpu_cleanup

    if stage == "temporal" and not cfg.temporal.enabled:
        log.warning("temporal.enabled=False, skipping")
        return {"status": "skipped", "reason": "temporal.enabled=False"}

    pl.seed_everything(cfg.seed, workers=True)
    dm, device = _build_dm(cfg, stage)
    module = _build_module(cfg, stage, device, dm)

    overrides = module.trainer_overrides(cfg, dm) if hasattr(module, "trainer_overrides") else {}
    trainer = _make_trainer(cfg, **overrides)

    trainer.fit(module, datamodule=dm, ckpt_path=os.environ.get("KD_GAT_CKPT_PATH"))

    ckpt = getattr(trainer.checkpoint_callback, "best_model_path", "")
    metrics = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}
    log.info("training_complete", stage=stage, checkpoint=ckpt)
    gpu_cleanup()
    return {"checkpoint": ckpt, "metrics": metrics}


def _make_trainer(cfg, **overrides) -> pl.Trainer:
    """pl.Trainer with cfg-driven defaults. SLURM auto-detected by Lightning."""
    if "callbacks" not in overrides:
        t = cfg.training
        overrides["callbacks"] = [
            ModelCheckpoint(
                dirpath=".", filename="best_model",
                monitor=t.monitor_metric, mode=t.monitor_mode,
                save_top_k=t.save_top_k, save_on_train_epoch_end=False,
            ),
            EarlyStopping(
                monitor=t.monitor_metric, patience=t.patience,
                mode=t.monitor_mode, check_on_train_epoch_end=False,
            ),
            DeviceStatsMonitor(cpu_stats=True),
            LearningRateMonitor(logging_interval="step"),
            StochasticWeightAveraging(swa_lrs=0.001, swa_epoch_start=0.75),
        ]

    kwargs = dict(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=cfg.training.gradient_clip,
        precision=cfg.training.precision,
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        deterministic=cfg.training.deterministic,
        benchmark=cfg.training.cudnn_benchmark,
        enable_progress_bar=not bool(os.environ.get("SLURM_JOB_ID")),
    )
    kwargs.update(overrides)
    return pl.Trainer(**kwargs)


def _build_dm(cfg, stage: str):
    """Dispatch to the right DataModule. Returns (dm, device)."""
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

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
    raw_dm = CANBusDataModule.from_cfg(cfg)
    raw_dm.setup("fit")
    raw_dm.populate_config(cfg)

    if stage == "curriculum":
        from graphids.core.preprocessing.curriculum import CurriculumDataModule
        dm = CurriculumDataModule.from_cfg(cfg, raw_dm, load_model_fn=load_model)
        return dm, device

    return raw_dm, device


def _build_module(cfg, stage: str, device, dm=None):
    """Dispatch to the right LightningModule."""
    if stage == "fusion":
        from graphids.core.models.fusion_baselines import build_fusion_module
        return build_fusion_module(cfg, device)
    if stage == "temporal":
        from graphids.core.models.temporal import TemporalLightningModule
        return TemporalLightningModule.from_datamodule(cfg, dm)

    from graphids.core.models._training import prepare_kd
    from graphids.core.models.registry import get_module_cls

    module_cls = get_module_cls(cfg.model_type)
    teacher, projection = prepare_kd(cfg, cfg.model_type, device)
    return module_cls(cfg, teacher=teacher, projection=projection)


# ---------------------------------------------------------------------------
# Evaluation — pl.Trainer.test() in a loop
# ---------------------------------------------------------------------------

_EVAL_ORDER = ["gat", "vgae", "dgi", "fusion", "temporal"]


def _evaluate(cfg) -> dict:
    """Evaluate trained models: Trainer.test() per model + artifact generation."""
    from graphids.core.models._training import eval_with_scenarios, gpu_cleanup, test_model
    from graphids.core.models.fusion_baselines import run_fusion_inference
    from graphids.core.models.registry import get_module_cls
    from graphids.core.preprocessing import CANBusDataModule, FusionDataModule

    pl.seed_everything(cfg.seed, workers=True)
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


# ---------------------------------------------------------------------------
# DuckDB catalog
# ---------------------------------------------------------------------------

_CATALOG_SCHEMA = (
    "run_dir VARCHAR, dataset VARCHAR, model_type VARCHAR, scale VARCHAR, "
    "stage VARCHAR, seed BIGINT, created_at TIMESTAMP DEFAULT current_timestamp, "
    "slurm_job_id VARCHAR, identity_hash VARCHAR, config_name VARCHAR, "
    "config JSON, metric_val_loss DOUBLE, metric_train_loss DOUBLE, "
    "metric_val_acc DOUBLE, metric_train_acc DOUBLE"
)


def _append_to_catalog(cfg, stage: str, result: dict, run_dir: Path) -> None:
    """Append run result to DuckDB catalog. Best-effort — never fails the job."""
    try:
        import duckdb

        from graphids.config import compute_identity_hash

        catalog_path = Path(cfg.lake_root) / "catalog" / "kd_gat.duckdb"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        db = duckdb.connect(str(catalog_path))
        db.execute(f"CREATE TABLE IF NOT EXISTS experiments ({_CATALOG_SCHEMA})")

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        identity_hash = compute_identity_hash(stage, cfg).lstrip("_") or None
        db.execute(
            """INSERT INTO experiments (
                run_dir, dataset, model_type, scale, stage, seed,
                slurm_job_id, identity_hash, config, config_name,
                metric_val_loss, metric_train_loss, metric_val_acc, metric_train_acc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                str(run_dir), cfg.dataset, cfg.model_type, cfg.scale, stage, cfg.seed,
                os.environ.get("SLURM_JOB_ID", ""),
                identity_hash, json.dumps(cfg.as_dict()),
                os.environ.get("KD_GAT_CONFIG_NAME", ""),
                metrics.get("val_loss"), metrics.get("train_loss"),
                metrics.get("val_acc"), metrics.get("train_acc"),
            ],
        )
        db.close()
        log.info("catalog_appended", catalog=str(catalog_path))
    except Exception as e:
        log.warning("catalog_append_failed", error=str(e))
