"""All paths derived from PipelineConfig. One function, one truth.

Every file location in the entire system comes from stage_dir().
The CLI, the stages -- they all call these functions.
No second implementation. No disagreement possible.

Path layout: {root}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]

Two write/read interfaces:
  - PipelineConfig-based (stage_dir, checkpoint_path, etc.) -- used by Python stages
  - String-based (_str variants) -- convenience for raw-string path construction

Artifact resolution:
  - ArtifactResolver -- cache-first, MLflow-fallback for cross-stage reads
  - run_group() / run_metadata() -- seed-independent identity for aggregation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import (  # noqa: F401  # re-exported via config/__init__
    CATALOG_PATH,
    STAGES,
)

if TYPE_CHECKING:
    from .schema import PipelineConfig

log = logging.getLogger(__name__)

EXPERIMENT_ROOT = os.environ.get("KD_GAT_EXPERIMENT_ROOT", "experimentruns")

# MLflow tracking URI — defaults to SQLite in data/mlflow/
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{Path(__file__).resolve().parents[2] / 'data' / 'mlflow' / 'mlflow.db'}",
)

# Data storage root — overridable via env var for easy migration between
# home dir (dev) and project storage (production).
_DATA_ROOT: str | None = os.environ.get("KD_GAT_DATA_ROOT")
_CACHE_ROOT: str | None = os.environ.get("KD_GAT_CACHE_ROOT")

# Artifact cache root — disposable, rm -rf safe
_ARTIFACT_CACHE_ROOT: str = os.environ.get("KD_GAT_ARTIFACT_CACHE", ".cache/kd-gat")

# ESS data lake root — when set, lake paths take priority
_LAKE_ROOT: str | None = os.environ.get("KD_GAT_LAKE_ROOT")


def _lake_run_dir(
    lake_root: Path,
    dataset: str,
    model_type: str,
    scale: str,
    stage: str,
    aux: str = "",
    seed: int = 42,
) -> Path:
    """Derive lake run directory from identity dimensions (no lake module import).

    Defaults to dev/{user}/ unless KD_GAT_PRODUCTION=1.
    """
    import getpass

    production = os.environ.get("KD_GAT_PRODUCTION", "").lower() in ("1", "true")
    tier = "production" if production else f"dev/{getpass.getuser()}"
    model = "eval" if stage == "evaluation" else model_type
    suffix = f"_{aux}" if aux else ""
    return lake_root / tier / dataset / f"{model}_{scale}_{stage}{suffix}" / f"seed_{seed}"


_datasets_cache: list[str] | None = None


def get_datasets() -> list[str]:
    """Read dataset names from config/datasets.yaml (cached after first call).

    Uses Pydantic-validated catalog loader for early error detection.
    """
    global _datasets_cache
    if _datasets_cache is None:
        from .catalog import load_catalog

        catalog = load_catalog()
        _datasets_cache = list(catalog.keys())
    return _datasets_cache


# ---------------------------------------------------------------------------
# Run identity helpers
# ---------------------------------------------------------------------------


def _run_id_parts(
    cfg: PipelineConfig,
    stage: str | None = None,
) -> tuple[str, str, str]:
    """Extract the three variable parts of a run ID from config.

    Returns ``(model_type, scale, aux_suffix)`` where *model_type* is
    overridden to ``"eval"`` for the evaluation stage and *aux_suffix*
    is the underscore-prefixed auxiliary type (or ``""``).
    """
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    model = "eval" if stage == "evaluation" else cfg.model_type
    return model, cfg.scale, aux_suffix


def run_id(cfg: PipelineConfig, stage: str) -> str:
    """Deterministic run ID from config and stage.

    Format: {dataset}/{model_type}_{scale}_{stage}[_{aux}]
    Examples:
        - "hcrl_sa/vgae_large_autoencoder"
        - "hcrl_sa/gat_small_curriculum_kd"
        - "set_01/dqn_large_fusion"

    This ID is used for:
    - Filesystem directory names (via stage_dir)
    - MLflow run names (for tracking)
    """
    model, scale, aux_suffix = _run_id_parts(cfg, stage)
    return f"{cfg.dataset}/{model}_{scale}_{stage}{aux_suffix}"


#: Seed-independent run identity for aggregation across seeds.
#: Same as :func:`run_id` — the seed is NOT part of the group key.
#: Used to query MLflow for all seeds of a given configuration.
run_group = run_id


def run_metadata(cfg: PipelineConfig, stage: str) -> dict[str, str]:
    """Single source of truth for all MLflow tags on a run.

    Every run gets these tags. They serve as the run's identity in MLflow
    and are used by ArtifactResolver to find runs.
    """
    return {
        "dataset": cfg.dataset,
        "model_type": cfg.model_type,
        "scale": cfg.scale,
        "stage": stage,
        "auxiliaries": cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
        "seed": str(cfg.seed),
        "run_group": run_group(cfg, stage),
        "config_hash": _config_hash(cfg),
    }


def _config_hash(cfg: PipelineConfig) -> str:
    """Deterministic short hash of config for deduplication."""
    return hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Write paths (for current stage output)
# ---------------------------------------------------------------------------


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical experiment directory for writing stage output.

    Resolution order:
    1. KD_GAT_STAGE_DIR (node-local SSD in SLURM jobs)
    2. Lake root (if KD_GAT_LAKE_ROOT set, writes to dev/{user}/ by default)
    3. experiment_root (legacy, in-repo experimentruns/)

    When writing to the lake, adds seed_{N}/ subdirectory.
    """
    stage_root = os.environ.get("KD_GAT_STAGE_DIR")
    if stage_root:
        return Path(stage_root) / run_id(cfg, stage) / f"seed_{cfg.seed}"

    if _LAKE_ROOT:
        aux = cfg.auxiliaries[0].type if cfg.auxiliaries else ""
        return _lake_run_dir(
            Path(_LAKE_ROOT),
            cfg.dataset,
            cfg.model_type,
            cfg.scale,
            stage,
            aux=aux,
            seed=cfg.seed,
        )

    # Legacy: no lake, no STAGE_DIR
    return Path(cfg.experiment_root) / run_id(cfg, stage) / f"seed_{cfg.seed}"


def checkpoint_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the best model checkpoint is saved."""
    return stage_dir(cfg, stage) / "best_model.pt"


def config_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the frozen config JSON is saved alongside the model."""
    return stage_dir(cfg, stage) / "config.json"


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset.

    Resolution order:
    1. KD_GAT_DATA_ROOT/raw/{dataset} (explicit override)
    2. Lake root/raw/{dataset} (ESS data lake)
    3. data/automotive/{dataset} (in-repo legacy)
    """
    if _DATA_ROOT:
        candidate = Path(_DATA_ROOT) / "raw" / cfg.dataset
        if candidate.exists():
            return candidate

    if _LAKE_ROOT:
        candidate = Path(_LAKE_ROOT) / "raw" / cfg.dataset
        if candidate.exists():
            return candidate

    return Path("data") / "automotive" / cfg.dataset


def metrics_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the evaluation metrics JSON is saved."""
    return stage_dir(cfg, stage) / "metrics.json"


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory.

    Priority: KD_GAT_CACHE_ROOT > Lake root > KD_GAT_DATA_ROOT/cache > data/cache (in-repo).
    """
    if _CACHE_ROOT:
        return Path(_CACHE_ROOT) / cfg.dataset

    if _LAKE_ROOT:
        from .constants import PREPROCESSING_VERSION

        return Path(_LAKE_ROOT) / "cache" / f"v{PREPROCESSING_VERSION}" / cfg.dataset

    if _DATA_ROOT:
        return Path(_DATA_ROOT) / "cache" / cfg.dataset
    return Path("data") / "cache" / cfg.dataset


# ---------------------------------------------------------------------------
# String-based path functions (no PipelineConfig needed)
# ---------------------------------------------------------------------------


def run_id_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Deterministic run ID from raw strings."""
    suffix = f"_{aux}" if aux else ""
    model = "eval" if stage == "evaluation" else model_type
    return f"{dataset}/{model}_{scale}_{stage}{suffix}"


def checkpoint_path_str(
    dataset: str, model_type: str, scale: str, stage: str, aux: str = "", seed: int = 42
) -> str:
    """Checkpoint path from raw strings (with seed subdirectory)."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/seed_{seed}/best_model.pt"


def metrics_path_str(
    dataset: str, model_type: str, scale: str, stage: str, aux: str = "", seed: int = 42
) -> str:
    """Metrics JSON path from raw strings (with seed subdirectory)."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/seed_{seed}/metrics.json"


# ---------------------------------------------------------------------------
# ArtifactResolver — cache-first, MLflow-fallback for cross-stage reads
# ---------------------------------------------------------------------------


class ArtifactResolver:
    """Resolves artifact locations: cache-first, MLflow-fallback.

    Used for loading artifacts from OTHER stages (e.g. loading VGAE checkpoint
    while training GAT). Same-stage writes use stage_dir() directly.

    Cache layout: .cache/kd-gat/{run_group}/seed_{seed}/{artifact_name}
    """

    def __init__(self, cache_root: Path | None = None):
        self.cache_root = Path(cache_root or _ARTIFACT_CACHE_ROOT)
        self._client = None  # lazy MlflowClient

    @property
    def client(self):
        if self._client is None:
            from mlflow import MlflowClient

            self._client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        return self._client

    def get(
        self,
        cfg: PipelineConfig,
        stage: str,
        artifact_name: str,
        model_type: str | None = None,
    ) -> Path:
        """Get artifact path. Downloads from MLflow if not cached.

        For cross-model reads (e.g. loading VGAE from GAT config), pass
        model_type to override cfg.model_type.
        """
        # Build the effective config identity for this artifact
        mt = model_type or cfg.model_type
        group = self._run_group_str(cfg, stage, mt)
        cache_path = self.cache_root / group / f"seed_{cfg.seed}" / artifact_name

        if cache_path.exists():
            return cache_path

        # Fall back to legacy experimentruns/ path (transitional)
        legacy_path = self._legacy_path(cfg, stage, artifact_name, mt)
        if legacy_path.exists():
            # Populate cache from legacy location
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_path, cache_path)
            log.debug("Cached legacy artifact: %s → %s", legacy_path, cache_path)
            return cache_path

        # Try MLflow download
        try:
            run = self._find_run(cfg, stage, mt)
            if run is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_artifacts(
                    run.info.run_id, artifact_name, str(cache_path.parent)
                )
                if cache_path.exists():
                    log.info("Downloaded from MLflow: %s", cache_path)
                    return cache_path
        except Exception as e:
            log.debug("MLflow download failed for %s/%s: %s", group, artifact_name, e)

        raise FileNotFoundError(
            f"Artifact not found: {artifact_name} for {group}/seed_{cfg.seed}. "
            f"Checked: cache ({cache_path}), legacy ({legacy_path}), MLflow."
        )

    def put(
        self,
        cfg: PipelineConfig,
        stage: str,
        local_path: Path,
    ) -> None:
        """Log artifact to MLflow and populate cache.

        Called after training writes artifacts to stage_dir().
        """
        import mlflow

        if local_path.exists():
            mlflow.log_artifact(str(local_path))

            # Populate cache
            group = run_group(cfg, stage)
            cache_dest = self.cache_root / group / f"seed_{cfg.seed}" / local_path.name
            cache_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, cache_dest)
            log.debug("Cached artifact: %s → %s", local_path, cache_dest)

    def exists(
        self,
        cfg: PipelineConfig,
        stage: str,
        artifact_name: str,
        model_type: str | None = None,
    ) -> bool:
        """Check if an artifact exists without downloading."""
        mt = model_type or cfg.model_type
        group = self._run_group_str(cfg, stage, mt)
        cache_path = self.cache_root / group / f"seed_{cfg.seed}" / artifact_name
        if cache_path.exists():
            return True

        legacy_path = self._legacy_path(cfg, stage, artifact_name, mt)
        return legacy_path.exists()

    def _run_group_str(self, cfg: PipelineConfig, stage: str, model_type: str) -> str:
        """Build run group string, possibly with overridden model_type."""
        _, scale, aux_suffix = _run_id_parts(cfg, stage)
        model = "eval" if stage == "evaluation" else model_type
        return f"{cfg.dataset}/{model}_{scale}_{stage}{aux_suffix}"

    def _legacy_path(
        self, cfg: PipelineConfig, stage: str, artifact_name: str, model_type: str
    ) -> Path:
        """Build experimentruns/ path, checking seed subdir then flat layout."""
        _, scale, aux_suffix = _run_id_parts(cfg, stage)
        base = Path(cfg.experiment_root) / cfg.dataset / f"{model_type}_{scale}_{stage}{aux_suffix}"
        # New layout: seed subdirectory
        seed_path = base / f"seed_{cfg.seed}" / artifact_name
        if seed_path.exists():
            return seed_path
        # Legacy: flat (no seed subdirectory)
        return base / artifact_name

    def _find_run(self, cfg: PipelineConfig, stage: str, model_type: str):
        """Find MLflow run by tags. Returns None if not found."""
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
            runs = self.client.search_runs(
                experiment_ids=[],  # search all experiments
                filter_string=" AND ".join(filter_parts),
                max_results=1,
                order_by=["start_time DESC"],
            )
            return runs[0] if runs else None
        except Exception:
            return None


# Module-level resolver singleton
_resolver: ArtifactResolver | None = None


def get_resolver() -> ArtifactResolver:
    """Get the module-level ArtifactResolver singleton."""
    global _resolver
    if _resolver is None:
        _resolver = ArtifactResolver()
    return _resolver
