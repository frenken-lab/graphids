"""Configuration layer: loads YAML, derives topology, path helpers."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Static constants (from constants.yaml)
# ---------------------------------------------------------------------------
_constants = yaml.safe_load((CONFIG_DIR / "constants.yaml").read_text())

PREPROCESSING_VERSION: str = _constants["preprocessing_version"]
MAX_DATA_BYTES: int = _constants["max_data_bytes"]
EXCLUDED_ATTACK_TYPES: list[str] = _constants["excluded_attack_types"]

# ---------------------------------------------------------------------------
# SLURM env vars (fallback to YAML defaults)
# ---------------------------------------------------------------------------
_slurm = _constants["slurm"]
SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", _slurm["account"])
SLURM_LOG_DIR: str = os.environ.get("KD_GAT_SLURM_LOG_DIR", _slurm["log_dir"])

# ---------------------------------------------------------------------------
# Lake root — base for all experiment IO (expanded configs, run dirs, catalog)
# ---------------------------------------------------------------------------
LAKE_ROOT: str = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")

# ---------------------------------------------------------------------------
# Write paths — loaded from write_paths.yaml (single source of truth)
# ---------------------------------------------------------------------------
_write_paths = yaml.safe_load((CONFIG_DIR / "write_paths.yaml").read_text())

CKPT_SUBPATH: str = _write_paths["lightning"]["checkpoint"]
LAST_CKPT_SUBPATH: str = _write_paths["lightning"]["last_checkpoint"]
COMPLETE_MARKER: str = _write_paths["dagster"]["complete_marker"]
DAGSTER_IO_DIR_TEMPLATE: str = _write_paths["dagster"]["io_dir"]
DAGSTER_HOME_DEFAULT: str = _write_paths["dagster"]["home"]
WANDB_WRITE_DIR: str = os.environ.get("WANDB_DIR", _write_paths["wandb"]["dir"])


def run_dir(lake_root: str, user: str, dataset: str, model_type: str,
            scale: str, stage: str, identity: str, kd_tag: str, seed: int) -> str:
    """Deterministic run directory path. Used by dagster + SLURM --trainer.default_root_dir."""
    return (f"{lake_root}/dev/{user}/{dataset}"
            f"/{model_type}_{scale}_{stage}{identity}{kd_tag}/seed_{seed}")


SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", _slurm["partition"])
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", _slurm["gpu_type"])
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")

# ---------------------------------------------------------------------------
# Pipeline topology (derived from defaults/pipeline.yaml)
# ---------------------------------------------------------------------------
PIPELINE_YAML: dict = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())

STAGES: dict[str, tuple[str, str, str]] = {
    name: (s["learning_type"], s["model"], s["mode"])
    for name, s in PIPELINE_YAML["stages"].items()
}
STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    name: [(d["model"], d["stage"]) for d in s.get("depends_on", [])]
    for name, s in PIPELINE_YAML["stages"].items()
    if s.get("depends_on")
}
VALID_MODEL_TYPES: frozenset[str] = frozenset(PIPELINE_YAML["models"])
VALID_SCALES: frozenset[str] = frozenset(PIPELINE_YAML["scales"])

# Checkpoint stage mapping — lives in pipeline.yaml (co-located with topology).
_CKPT_STAGES: dict[str, str] = PIPELINE_YAML["ckpt_stages"]
_missing_ckpt = set(PIPELINE_YAML["models"]) - set(_CKPT_STAGES.keys())
if _missing_ckpt:
    raise ValueError(
        f"Models {_missing_ckpt} in pipeline.yaml 'models' missing from 'ckpt_stages'. "
        f"Add entries to ckpt_stages in pipeline.yaml."
    )

DEFAULT_MODEL_TYPE: str = next(iter(PIPELINE_YAML["models"]))
DEFAULT_SCALE: str = PIPELINE_YAML["scales"][0]
DEFAULT_STAGE: str = PIPELINE_YAML["default_stages"][0]
CATALOG_PATH: Path = CONFIG_DIR / "datasets.yaml"
_datasets = yaml.safe_load(CATALOG_PATH.read_text())
DEFAULT_DATASET: str = next(k for k in _datasets if not k.startswith("_"))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def compute_preprocessing_hash() -> str:
    """Content-addressable hash of preprocessing parameters."""
    import hashlib

    from graphids.core.preprocessing.features import N_EDGE_FEATURES, N_NODE_FEATURES

    components = [PREPROCESSING_VERSION, str(N_NODE_FEATURES), str(N_EDGE_FEATURES), "100", "100", "0.8"]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]


def data_dir(lake_root: str, dataset: str) -> Path:
    """Raw data directory. Tries lake, falls back to local."""
    candidate = Path(lake_root) / "raw" / dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    """Processed-graph cache directory."""
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


def compute_identity_hash(stage: str, cfg) -> str:
    """Compute identity hash for a stage from its identity_keys.

    Returns ``"_<8-char-hex>"`` or ``""`` if the stage has no identity keys.
    """
    import hashlib

    stage_def = PIPELINE_YAML.get("stages", {}).get(stage, {})
    keys = stage_def.get("identity_keys", [])
    if not keys:
        return ""

    def _get(dotted_key, default=None):
        cur = cfg
        for part in dotted_key.split("."):
            if cur is None:
                return default
            cur = cur.get(part) if isinstance(cur, dict) else getattr(cur, part, None)
        return cur if cur is not None else default

    unresolved = [k for k in keys if _get(k) is None]
    if unresolved:
        raise KeyError(
            f"Identity keys {unresolved} not found in config for stage '{stage}'. "
            f"These keys must be set in the YAML config or model __init__ for correct "
            f"checkpoint path computation."
        )
    pairs = [f"{k}={_get(k, '_default_')}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]


def checkpoint_path(lake_root: str, dataset: str, model_type: str, scale: str,
                    seed: int, cfg, *, gat_stage: str = "curriculum") -> Path:
    """Compute the expected checkpoint path for a trained model."""
    user = os.environ.get("USER", "unknown")
    output_base = f"{lake_root}/dev/{user}/{dataset}"
    stage = _CKPT_STAGES.get(model_type, model_type)
    if model_type == "gat":
        stage = gat_stage
    identity = compute_identity_hash(stage, cfg)
    return Path(f"{output_base}/{model_type}_{scale}_{stage}{identity}/seed_{seed}/{CKPT_SUBPATH}")
