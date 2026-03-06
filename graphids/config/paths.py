"""All paths derived from PipelineConfig. One function, one truth.

Every file location in the entire system comes from stage_dir().
The CLI, the stages -- they all call these functions.
No second implementation. No disagreement possible.

Path layout: {root}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]

Two interfaces:
  - PipelineConfig-based (stage_dir, checkpoint_path, etc.) -- used by Python stages
  - String-based (_str variants) -- convenience for raw-string path construction
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import PipelineConfig

EXPERIMENT_ROOT = os.environ.get("KD_GAT_EXPERIMENT_ROOT", "experimentruns")

# Data storage root — overridable via env var for easy migration between
# home dir (dev) and project storage (production).
_DATA_ROOT: str | None = os.environ.get("KD_GAT_DATA_ROOT")
_CACHE_ROOT: str | None = os.environ.get("KD_GAT_CACHE_ROOT")

# stage_name -> (learning_type, model_arch, training_mode)
# run_id() overrides model_arch to "eval" for the evaluation stage.
STAGES = {
    "autoencoder": ("unsupervised", "vgae", "autoencoder"),
    "curriculum": ("supervised", "gat", "curriculum"),
    "normal": ("supervised", "gat", "normal"),
    "fusion": ("rl_fusion", "dqn", "fusion"),
    "evaluation": ("evaluation", "eval", "evaluation"),
    "temporal": ("temporal", "gat", "temporal"),
}

from .constants import CATALOG_PATH  # noqa: F401  # re-exported via config/__init__

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


# Backwards-compatible module-level name (lazy property via __getattr__)
def __getattr__(name: str):
    if name == "DATASETS":
        return get_datasets()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_id(cfg: PipelineConfig, stage: str) -> str:
    """Deterministic run ID from config and stage.

    Format: {dataset}/{model_type}_{scale}_{stage}[_{aux}]
    Examples:
        - "hcrl_sa/vgae_large_autoencoder"
        - "hcrl_sa/gat_small_curriculum_kd"
        - "set_01/dqn_large_fusion"

    This ID is used for:
    - Filesystem directory names (via stage_dir)
    - W&B run names (for tracking)
    """
    aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
    model = "eval" if stage == "evaluation" else cfg.model_type
    return f"{cfg.dataset}/{model}_{cfg.scale}_{stage}{aux_suffix}"


def stage_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Canonical experiment directory.

    Layout: {root}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]
    """
    return Path(cfg.experiment_root) / run_id(cfg, stage)


def checkpoint_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the best model checkpoint is saved."""
    return stage_dir(cfg, stage) / "best_model.pt"


def config_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the frozen config JSON is saved alongside the model."""
    return stage_dir(cfg, stage) / "config.json"


def log_dir(cfg: PipelineConfig, stage: str) -> Path:
    """Lightning / CSV log directory for a stage."""
    return stage_dir(cfg, stage) / "logs"


def data_dir(cfg: PipelineConfig) -> Path:
    """Raw data directory for a dataset.

    When KD_GAT_DATA_ROOT is set, looks for raw data under
    ``$KD_GAT_DATA_ROOT/raw/{dataset}``, falling back to the in-repo
    ``data/automotive/{dataset}`` for backwards compatibility.
    """
    if _DATA_ROOT:
        candidate = Path(_DATA_ROOT) / "raw" / cfg.dataset
        if candidate.exists():
            return candidate
    return Path("data") / "automotive" / cfg.dataset


def metrics_path(cfg: PipelineConfig, stage: str) -> Path:
    """Where the evaluation metrics JSON is saved."""
    return stage_dir(cfg, stage) / "metrics.json"


def cache_dir(cfg: PipelineConfig) -> Path:
    """Processed-graph cache directory.

    Priority: KD_GAT_CACHE_ROOT > KD_GAT_DATA_ROOT/cache > data/cache (in-repo).
    """
    if _CACHE_ROOT:
        return Path(_CACHE_ROOT) / cfg.dataset
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
    dataset: str, model_type: str, scale: str, stage: str, aux: str = ""
) -> str:
    """Checkpoint path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/best_model.pt"


def metrics_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Metrics JSON path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/metrics.json"


def benchmark_path_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
    """Benchmark TSV path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/benchmark.tsv"


def log_path_str(
    dataset: str, model_type: str, scale: str, stage: str, aux: str = "", stream: str = "out"
) -> str:
    """SLURM log path from raw strings."""
    return f"{EXPERIMENT_ROOT}/{run_id_str(dataset, model_type, scale, stage, aux)}/slurm.{stream}"
