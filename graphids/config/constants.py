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

# ---------------------------------------------------------------------------
# Project defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET = "hcrl_sa"
DEFAULT_LAKE_ROOT = "experimentruns"
DEFAULT_SEEDS = [42, 123, 456]
SWEEP_RESULTS_DIR = "data/sweep_results"

# ---------------------------------------------------------------------------
# Pipeline topology (cached loader)
# ---------------------------------------------------------------------------
_pipeline_cache: dict | None = None


def load_pipeline_yaml() -> dict:
    """Load and cache pipeline.yaml."""
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    return _pipeline_cache


def _load_topology() -> (
    tuple[
        dict[str, tuple[str, str, str]],
        dict[str, str],
        frozenset[str],
        frozenset[str],
        dict[str, list[tuple[str, str]]],
    ]
):
    """Derive pipeline topology constants from pipeline.yaml."""
    pipeline = load_pipeline_yaml()

    stages: dict[str, tuple[str, str, str]] = {
        name: (s["learning_type"], s["model"], s["mode"]) for name, s in pipeline["stages"].items()
    }
    stage_model_map: dict[str, str] = {k: v[1] for k, v in stages.items()}
    valid_model_types: frozenset[str] = frozenset(pipeline["models"].keys())
    valid_scales: frozenset[str] = frozenset(pipeline["scales"])

    deps: dict[str, list[tuple[str, str]]] = {}
    for name, s in pipeline["stages"].items():
        dep_list = s.get("depends_on", [])
        if dep_list:
            deps[name] = [(d["model"], d["stage"]) for d in dep_list]

    return stages, stage_model_map, valid_model_types, valid_scales, deps


# Eagerly computed at import time (pipeline.yaml is small and always needed)
STAGES, STAGE_MODEL_MAP, VALID_MODEL_TYPES, VALID_SCALES, STAGE_DEPENDENCIES = _load_topology()
