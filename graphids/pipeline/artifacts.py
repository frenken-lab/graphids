"""Artifact store: cache-first lookup with filesystem and MLflow fallback.

Provides get/put/exists for cross-stage artifact resolution (e.g. loading
a VGAE checkpoint while training GAT). Same-stage writes use stage_dir().
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from graphids.config.schema import PipelineConfig

log = logging.getLogger(__name__)

_ARTIFACT_CACHE: str = os.environ.get("KD_GAT_ARTIFACT_CACHE", ".cache/kd-gat")
_mlflow_client = None


def _get_mlflow():
    global _mlflow_client
    if _mlflow_client is None:
        from mlflow import MlflowClient

        from graphids.config import MLFLOW_TRACKING_URI

        _mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    return _mlflow_client


def _id_parts(cfg: PipelineConfig, stage: str) -> tuple[str, str, str]:
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    model = "eval" if stage == "evaluation" else cfg.model_type
    return model, cfg.scale, aux_suffix


def _artifact_group(cfg: PipelineConfig, stage: str, model_type: str) -> str:
    _, scale, aux_suffix = _id_parts(cfg, stage)
    model = "eval" if stage == "evaluation" else model_type
    return f"{cfg.dataset}/{model}_{scale}_{stage}{aux_suffix}"


def _fs_artifact_path(cfg: PipelineConfig, stage: str, artifact_name: str, model_type: str) -> Path:
    """Check experimentruns/ path with seed subdirectory."""
    _, scale, aux_suffix = _id_parts(cfg, stage)
    model = "eval" if stage == "evaluation" else model_type
    base = Path(cfg.experiment_root) / cfg.dataset / f"{model}_{scale}_{stage}{aux_suffix}"
    return base / f"seed_{cfg.seed}" / artifact_name


def get_artifact(
    cfg: PipelineConfig,
    stage: str,
    artifact_name: str,
    model_type: str | None = None,
) -> Path:
    """Get artifact path. Downloads from MLflow if not cached.

    For cross-model reads (e.g. loading VGAE from GAT config), pass
    model_type to override cfg.model_type.
    """
    mt = model_type or cfg.model_type
    group = _artifact_group(cfg, stage, mt)
    cache_path = Path(_ARTIFACT_CACHE) / group / f"seed_{cfg.seed}" / artifact_name

    if cache_path.exists():
        return cache_path

    # Check experimentruns / stage_dir
    fs_path = _fs_artifact_path(cfg, stage, artifact_name, mt)
    if fs_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fs_path, cache_path)
        log.debug("Cached artifact: %s → %s", fs_path, cache_path)
        return cache_path

    # MLflow fallback
    try:
        run = _find_mlflow_run(cfg, stage, mt)
        if run is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _get_mlflow().download_artifacts(run.info.run_id, artifact_name, str(cache_path.parent))
            if cache_path.exists():
                log.info("Downloaded from MLflow: %s", cache_path)
                return cache_path
    except Exception as e:
        log.debug("MLflow download failed for %s/%s: %s", group, artifact_name, e)

    raise FileNotFoundError(
        f"Artifact not found: {artifact_name} for {group}/seed_{cfg.seed}. "
        f"Checked: cache ({cache_path}), filesystem ({fs_path}), MLflow."
    )


def put_artifact(cfg: PipelineConfig, stage: str, local_path: Path) -> None:
    """Log artifact to MLflow and populate cache."""
    import mlflow

    from graphids.config import run_id

    if local_path.exists():
        mlflow.log_artifact(str(local_path))
        group = run_id(cfg, stage)
        cache_dest = Path(_ARTIFACT_CACHE) / group / f"seed_{cfg.seed}" / local_path.name
        cache_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, cache_dest)
        log.debug("Cached artifact: %s → %s", local_path, cache_dest)


def artifact_exists(
    cfg: PipelineConfig,
    stage: str,
    artifact_name: str,
    model_type: str | None = None,
) -> bool:
    """Check if an artifact exists without downloading."""
    mt = model_type or cfg.model_type
    group = _artifact_group(cfg, stage, mt)
    cache_path = Path(_ARTIFACT_CACHE) / group / f"seed_{cfg.seed}" / artifact_name
    if cache_path.exists():
        return True
    return _fs_artifact_path(cfg, stage, artifact_name, mt).exists()


def _find_mlflow_run(cfg: PipelineConfig, stage: str, model_type: str):
    aux = cfg.auxiliaries[0].type if cfg.auxiliaries else "none"
    filter_parts = [
        f"tags.dataset = '{cfg.dataset}'",
        f"tags.model_type = '{model_type}'",
        f"tags.scale = '{cfg.scale}'",
        f"tags.stage = '{stage}'",
        f"tags.seed = '{cfg.seed}'",
    ]
    if aux != "none":
        filter_parts.append(f"tags.auxiliaries = '{aux}'")
    try:
        runs = _get_mlflow().search_runs(
            experiment_ids=[],
            filter_string=" AND ".join(filter_parts),
            max_results=1,
            order_by=["start_time DESC"],
        )
        return runs[0] if runs else None
    except Exception:
        return None
