"""Project-wide constants and pipeline topology loader.

No imports from other config submodules — this is the leaf dependency.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
DEFAULTS_DIR = CONFIG_DIR / "defaults"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = DEFAULTS_DIR / "datasets.yaml"

# ---------------------------------------------------------------------------
# Preprocessing constants
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "7.0.0"
MAX_DATA_BYTES = 8
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]


def compute_preprocessing_hash() -> str:
    """Content-addressable hash of preprocessing parameters."""
    import hashlib

    # Lazy import to avoid circular dependency (features.py → constants.py)
    from graphids.core.preprocessing.features import N_EDGE_FEATURES, N_NODE_FEATURES

    components = [PREPROCESSING_VERSION, str(N_NODE_FEATURES), str(N_EDGE_FEATURES), "100", "100", "0.8"]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Pipeline topology (derived from pipeline.yaml at import time)
# ---------------------------------------------------------------------------
_pipeline = yaml.safe_load((DEFAULTS_DIR / "pipeline.yaml").read_text())

STAGES: dict[str, tuple[str, str, str]] = {
    name: (s["learning_type"], s["model"], s["mode"])
    for name, s in _pipeline["stages"].items()
}
STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    name: [(d["model"], d["stage"]) for d in s.get("depends_on", [])]
    for name, s in _pipeline["stages"].items()
    if s.get("depends_on")
}
VALID_MODEL_TYPES: frozenset[str] = frozenset(_pipeline["models"])  # works for both list and dict
VALID_SCALES: frozenset[str] = frozenset(_pipeline["scales"])
# Full pipeline dict for orchestration (identity_keys, default_stages).
PIPELINE_YAML: dict = _pipeline

# Pipeline-derived defaults (so schema.py doesn't hardcode these)
DEFAULT_MODEL_TYPE: str = next(iter(_pipeline["models"]))  # first model
DEFAULT_SCALE: str = _pipeline["scales"][0]                 # first scale
DEFAULT_STAGE: str = _pipeline["default_stages"][0]         # first default stage
_datasets = yaml.safe_load((DEFAULTS_DIR / "datasets.yaml").read_text())
DEFAULT_DATASET: str = next(k for k in _datasets if not k.startswith("_"))  # skip YAML anchors

# ---------------------------------------------------------------------------
# Environment (KD_GAT_* infrastructure env vars)
# ---------------------------------------------------------------------------
SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def data_dir(lake_root: str, dataset: str) -> Path:
    """Raw data directory. Tries lake, falls back to local."""
    candidate = Path(lake_root) / "raw" / dataset
    if candidate.exists():
        return candidate
    return Path("data") / "automotive" / dataset


def cache_dir(lake_root: str, dataset: str) -> Path:
    """Processed-graph cache directory."""
    return Path(lake_root) / "cache" / f"v{PREPROCESSING_VERSION}" / dataset


# ---------------------------------------------------------------------------
# Identity hash
# ---------------------------------------------------------------------------


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
        import structlog
        structlog.get_logger().warning("identity_key_unresolved", stage=stage, keys=unresolved)
    pairs = [f"{k}={_get(k, '_default_')}" for k in sorted(keys)]
    return "_" + hashlib.sha256("|".join(pairs).encode()).hexdigest()[:8]
