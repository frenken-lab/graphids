"""Domain and infrastructure constants.

These are NOT hyperparameters (those live in PipelineConfig).
These are structural/environmental constants that rarely change.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "3.0.0"  # Collated tensor storage (was list[Data] in 2.x)

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
CATALOG_PATH = Path(__file__).parent / "datasets.yaml"

# Repository root (graphids/config/ is 2 levels deep)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]
DEFAULT_DATASET = "hcrl_sa"
MAX_DATA_BYTES = 8
NODE_FEATURE_COUNT = 26  # CAN_ID + 8 means + 8 stds + entropy + 2 change_rate + skew + kurt + clustering + split_half + count + position
EDGE_FEATURE_COUNT = 11  # Streamlined edge features


# ---------------------------------------------------------------------------
# Batch index utilities
# ---------------------------------------------------------------------------
def get_batch_index(g, device: "torch.device") -> "torch.Tensor":  # noqa: F821
    """Get batch index from graph, creating a single-graph default if absent."""
    import torch

    if hasattr(g, "batch") and g.batch is not None:
        return g.batch
    return torch.zeros(g.x.size(0), dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Attack type utilities (backward-compat for caches without attack_type)
# ---------------------------------------------------------------------------
def graph_attack_type(g, default: int | None = -1) -> int | None:
    """Get attack_type from a PyG graph, with backward-compat default.

    Old caches (pre-v2.0.0) lack the attack_type attribute.  This centralises
    the hasattr guard so callers don't scatter version-gating inline.
    """
    if hasattr(g, "attack_type") and g.attack_type is not None:
        return g.attack_type.item()
    return default


# ---------------------------------------------------------------------------
# DataLoader / memory mapping
# ---------------------------------------------------------------------------
# vm.max_map_count is typically 65530 on Linux.
# Both spawn workers and share_memory_() create mmap entries per tensor,
# so datasets exceeding this limit must use num_workers=0.
MMAP_TENSOR_LIMIT = 60000

# ---------------------------------------------------------------------------
# SLURM defaults (override via environment for cluster migration)
# ---------------------------------------------------------------------------

SLURM_ACCOUNT = os.getenv("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_PARTITION = os.getenv("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE = os.getenv("KD_GAT_GPU_TYPE", "v100")

# ---------------------------------------------------------------------------
# Stage definitions (single source of truth)
# ---------------------------------------------------------------------------
# stage_name -> (learning_type, model_arch, training_mode)
# run_id() overrides model_arch to "eval" for the evaluation stage.
STAGES: dict[str, tuple[str, str, str]] = {
    "autoencoder": ("unsupervised", "vgae", "autoencoder"),
    "curriculum": ("supervised", "gat", "curriculum"),
    "normal": ("supervised", "gat", "normal"),
    "fusion": ("rl_fusion", "dqn", "fusion"),
    "evaluation": ("evaluation", "eval", "evaluation"),
    "temporal": ("temporal", "gat", "temporal"),
}

# Derived: stage → model type (all stages including evaluation/temporal)
STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in STAGES.items()}

# Stage prerequisite dependencies: stage → list of (model_type, prereq_stage) pairs
STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = {
    "curriculum": [("vgae", "autoencoder")],
    "normal": [("vgae", "autoencoder")],
    "fusion": [("vgae", "autoencoder"), ("gat", "curriculum")],
    "temporal": [("gat", "curriculum")],
}

# ---------------------------------------------------------------------------
# Sweep / state output paths
# ---------------------------------------------------------------------------
SWEEP_RESULTS_DIR = "data/sweep_results"
SWEEP_STATE_DIR = "data/sweep_state"

# ---------------------------------------------------------------------------
# Multi-seed defaults (for statistical significance in TMLR submission)
# ---------------------------------------------------------------------------
DEFAULT_SEEDS: list[int] = [42, 123, 456]


def parse_seeds(value: str) -> list[int]:
    """Parse seeds: single int or comma-separated ints.

    Raises ValueError on invalid input (callers like argparse can wrap this).
    """
    if value is None:
        return []

    try:
        return [int(s.strip()) for s in value.split(",")]
    except ValueError as e:
        raise ValueError(f"Invalid seeds value '{value}': {e}") from e
