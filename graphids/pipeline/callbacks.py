"""Lightning callbacks for pipeline lifecycle: run directory, config persistence, catalog."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytorch_lightning as pl
import structlog
import yaml

log = structlog.get_logger()


class RunDirectorySetup(pl.Callback):
    """Create identity-hash run directory, save config + git SHA, bind structlog context.

    Fires in ``setup()`` (after dm.setup, before training starts).
    """

    def __init__(self, cfg, stage: str):
        self.cfg = cfg
        self.stage = stage

    def setup(self, trainer, pl_module, stage=None):
        from graphids.config import compute_identity_hash

        cfg = self.cfg
        identity = compute_identity_hash(self.stage, cfg)
        run_dir = (
            Path(cfg._output_base)
            / f"{cfg.model_type}_{cfg.scale}_{self.stage}{identity}"
            / f"seed_{cfg.seed}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(run_dir)

        # Persist config + git SHA
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

        # Structlog context for all subsequent log lines
        structlog.contextvars.bind_contextvars(
            dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
            stage=self.stage, seed=cfg.seed,
            slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
            run_dir=str(run_dir),
        )


class PopulateAndBuild(pl.Callback):
    """Wire data-derived dimensions into the model after dm.setup().

    Lightning calls ``dm.setup()`` before ``Callback.setup()``, so the
    DataModule's ``populate_config`` has access to dataset metadata (num_ids,
    in_channels, etc.). This callback reads those and calls
    ``pl_module.build_model()`` to construct the nn.Module with correct dims.
    """

    def setup(self, trainer, pl_module, stage=None):
        dm = trainer.datamodule
        if dm is not None and hasattr(dm, "populate_config") and hasattr(pl_module, "cfg"):
            dm.populate_config(pl_module.cfg)
        if hasattr(pl_module, "build_model") and pl_module.model is None:
            pl_module.build_model()


class DuckDBCatalog(pl.Callback):
    """Append run metadata + final metrics to DuckDB catalog. Best-effort."""

    _SCHEMA = (
        "run_dir VARCHAR, dataset VARCHAR, model_type VARCHAR, scale VARCHAR, "
        "stage VARCHAR, seed BIGINT, created_at TIMESTAMP DEFAULT current_timestamp, "
        "slurm_job_id VARCHAR, identity_hash VARCHAR, config_name VARCHAR, "
        "config JSON, metric_val_loss DOUBLE, metric_train_loss DOUBLE, "
        "metric_val_acc DOUBLE, metric_train_acc DOUBLE"
    )

    def __init__(self, cfg, stage: str):
        self.cfg = cfg
        self.stage = stage

    def teardown(self, trainer, pl_module, stage=None):
        try:
            import duckdb

            from graphids.config import compute_identity_hash

            cfg = self.cfg
            catalog_path = Path(cfg.lake_root) / "catalog" / "kd_gat.duckdb"
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            db = duckdb.connect(str(catalog_path))
            db.execute(f"CREATE TABLE IF NOT EXISTS experiments ({self._SCHEMA})")

            metrics = {
                k: v.item() if hasattr(v, "item") else v
                for k, v in trainer.callback_metrics.items()
            }
            identity_hash = compute_identity_hash(self.stage, cfg).lstrip("_") or None
            run_dir = Path.cwd()

            db.execute(
                """INSERT INTO experiments (
                    run_dir, dataset, model_type, scale, stage, seed,
                    slurm_job_id, identity_hash, config, config_name,
                    metric_val_loss, metric_train_loss, metric_val_acc, metric_train_acc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    str(run_dir), cfg.dataset, cfg.model_type, cfg.scale, self.stage, cfg.seed,
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
