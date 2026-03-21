"""Project-wide constants and pipeline topology loader.

No imports from other config submodules — this is the leaf dependency.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = CONFIG_DIR / "datasets.yaml"

# ---------------------------------------------------------------------------
# Preprocessing constants (change with feature engineering)
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "4.0.0"
MAX_DATA_BYTES = 8
NODE_FEATURE_COUNT = 26
EDGE_FEATURE_COUNT = 11
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]
MMAP_TENSOR_LIMIT = 60_000

# Preprocessing defaults (replaces PreprocessingConfig Pydantic model)
PREPROCESSING_DEFAULTS = {
    "window_size": 100,
    "stride": 100,
    "train_val_split": 0.8,
    "chunk_size": 5000,
    "ray_file_threshold": 4,
}


def compute_preprocessing_hash() -> str:
    """Content-addressable hash of preprocessing parameters."""
    import hashlib

    components = [
        PREPROCESSING_VERSION,
        str(NODE_FEATURE_COUNT),
        str(EDGE_FEATURE_COUNT),
        str(PREPROCESSING_DEFAULTS["window_size"]),
        str(PREPROCESSING_DEFAULTS["stride"]),
        str(PREPROCESSING_DEFAULTS["train_val_split"]),
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]

# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pipeline topology (derived from pipeline.yaml at import time)
# ---------------------------------------------------------------------------
_pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())

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
VALID_MODEL_TYPES: frozenset[str] = frozenset(_pipeline["models"])
VALID_SCALES: frozenset[str] = frozenset(_pipeline["scales"])
del _pipeline

# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "hcrl_sa"
DEFAULT_LAKE_ROOT = "experimentruns"
DEFAULT_SEEDS = [42, 123, 456]
SWEEP_RESULTS_DIR = "data/sweep_results"
