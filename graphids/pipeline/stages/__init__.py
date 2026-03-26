"""Pipeline stages: dispatch and run.

Public API:
    from graphids.pipeline.stages import run_stage
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

log = structlog.get_logger()

from .evaluation import evaluate
from .training import train_stage

STAGE_FNS = {
    "autoencoder": train_stage,
    "curriculum":  train_stage,
    "normal":      train_stage,
    "fusion":      train_stage,
    "evaluation":  evaluate,
    "temporal":    train_stage,
}


def run_stage(cfg, stage: str) -> dict:
    """Bind context, chdir to run directory, save config, run stage function.

    Output directory uses identity-aware paths via the identity_hash resolver:
      {lake_root}/{tier}/{dataset}/{model_type}_{scale}_{stage}_{hash}/seed_{seed}
    All Lightning outputs (checkpoints, logs, metrics) land in the data lake,
    not the project directory.
    """
    from omegaconf import OmegaConf
    from omegaconf.errors import ConfigAttributeError

    from graphids.config import STAGES

    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: {list(STAGES.keys())}")

    # Use the identity-aware path from Hydra config (resolved via identity_hash resolver)
    # When running via submitit, hydra.run.dir is pickled into cfg.
    # When running via `python -m graphids`, it's in HydraConfig instead.
    try:
        run_dir = Path(cfg.hydra.run.dir)
    except (AttributeError, ConfigAttributeError):
        from hydra.core.hydra_config import HydraConfig
        run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)

    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        run_dir=str(run_dir),
    )
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Capture git SHA (replaces RunMetadataCallback)
    import json
    import subprocess
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    (run_dir / "run_metadata.json").write_text(json.dumps({"git_sha": sha}, indent=2))

    result = STAGE_FNS[stage](cfg)
    _append_to_catalog(cfg, stage, result, run_dir)
    return result


def _append_to_catalog(cfg, stage: str, result: dict, run_dir: Path) -> None:
    """Append run result to DuckDB catalog. Best-effort — never fails the job."""
    try:
        import json

        import duckdb
        from omegaconf import OmegaConf

        catalog_path = Path(cfg.lake_root) / "catalog" / "kd_gat.duckdb"
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        db = duckdb.connect(str(catalog_path))
        _SCHEMA = {
            "run_dir": "VARCHAR", "tier": "VARCHAR", "dataset": "VARCHAR",
            "model_type": "VARCHAR", "scale": "VARCHAR", "stage": "VARCHAR",
            "auxiliaries": "VARCHAR", "seed": "BIGINT",
            "created_at": "TIMESTAMP DEFAULT current_timestamp",
            "graphids_version": "VARCHAR", "git_sha": "VARCHAR",
            "slurm_job_id": "VARCHAR", "num_artifacts": "BIGINT",
            "lr": "DOUBLE", "max_epochs": "BIGINT", "batch_size": "BIGINT",
            "precision": "VARCHAR", "has_kd": "BOOLEAN",
            "metric_val_loss": "DOUBLE", "metric_train_loss": "DOUBLE",
            "metric_epochs_run": "BIGINT",
            "metric_train_acc": "DOUBLE", "metric_val_acc": "DOUBLE",
            "identity_hash": "VARCHAR", "config_name": "VARCHAR",
            "config": "JSON", "identity_values": "VARCHAR",
        }
        cols = ", ".join(f"{k} {v}" for k, v in _SCHEMA.items())
        db.execute(f"CREATE TABLE IF NOT EXISTS experiments ({cols})")
        # Self-heal: add any columns missing from older databases
        existing = {r[0] for r in db.execute("SELECT column_name FROM information_schema.columns WHERE table_name='experiments'").fetchall()}
        for col, dtype in _SCHEMA.items():
            if col not in existing:
                db.execute(f"ALTER TABLE experiments ADD COLUMN {col} {dtype.split()[0]}")
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        # Resolve identity hash via the registered OmegaConf resolver (graphids.config)
        raw_hash = OmegaConf.create({"_h": f"${{identity_hash:{stage}}}"}, parent=cfg)._h
        identity_hash = raw_hash.lstrip("_") or None
        config_json = json.dumps(OmegaConf.to_container(cfg, resolve=True))
        db.execute(
            """INSERT INTO experiments (
                run_dir, dataset, model_type, scale, stage, seed,
                slurm_job_id, identity_hash, config, config_name,
                metric_val_loss, metric_train_loss, metric_val_acc, metric_train_acc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                str(run_dir), cfg.dataset, cfg.model_type, cfg.scale, stage, cfg.seed,
                os.environ.get("SLURM_JOB_ID", ""),
                identity_hash, config_json,
                os.environ.get("KD_GAT_CONFIG_NAME", ""),
                metrics.get("val_loss"), metrics.get("train_loss"),
                metrics.get("val_acc"), metrics.get("train_acc"),
            ],
        )
        db.close()
        log.info("catalog_appended", catalog=str(catalog_path))
    except Exception as e:
        log.warning("catalog_append_failed", error=str(e))
