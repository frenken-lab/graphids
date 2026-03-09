"""Domain and infrastructure constants.

These are NOT hyperparameters (those live in PipelineConfig).
These are structural/environmental constants that rarely change.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------
PREPROCESSING_VERSION = "2.2.0"  # Bump when graph construction logic changes

# ---------------------------------------------------------------------------
# Filesystem paths
# ---------------------------------------------------------------------------
CATALOG_PATH = Path(__file__).parent / "datasets.yaml"

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_SIZE = 100
DEFAULT_STRIDE = 100
EXCLUDED_ATTACK_TYPES = ["suppress", "masquerade"]
MAX_DATA_BYTES = 8
NODE_FEATURE_COUNT = 26  # CAN_ID + 8 means + 8 stds + entropy + 2 change_rate + skew + kurt + clustering + split_half + count + position
EDGE_FEATURE_COUNT = 11  # Streamlined edge features

# Ordered feature names matching engine.py computation order (for export/visualization)
NODE_FEATURE_NAMES: list[str] = [
    # [0] Entity ID
    "CAN_ID",
    # [1:9] Per-byte payload means
    "Byte0_Mean",
    "Byte1_Mean",
    "Byte2_Mean",
    "Byte3_Mean",
    "Byte4_Mean",
    "Byte5_Mean",
    "Byte6_Mean",
    "Byte7_Mean",
    # [9:17] Per-byte payload standard deviations
    "Byte0_Std",
    "Byte1_Std",
    "Byte2_Std",
    "Byte3_Std",
    "Byte4_Std",
    "Byte5_Std",
    "Byte6_Std",
    "Byte7_Std",
    # [17] Shannon entropy of byte values
    "Payload_Entropy",
    # [18:20] Payload change rates
    "Change_Rate_Mean",
    "Change_Rate_Max",
    # [20:22] Higher-order moments
    "Skewness",
    "Kurtosis",
    # [22] Local clustering coefficient
    "Clustering_Coeff",
    # [23] Split-half ratio
    "Split_Half_Ratio",
    # [24:26] Occurrence stats
    "Occurrence_Count",
    "Last_Position",
]

EDGE_FEATURE_NAMES: list[str] = [
    "Count",  # [0] Raw transition count
    "Frequency",  # [1] Relative frequency (count/window)
    "Mean_Interval",  # [2] Mean inter-arrival interval
    "Std_Interval",  # [3] Std inter-arrival interval
    "Regularity",  # [4] 1/(1+std)
    "First_Position",  # [5] First occurrence (normalized)
    "Last_Position",  # [6] Last occurrence (normalized)
    "Temporal_Span",  # [7] Last - first (normalized)
    "Bidirectional",  # [8] Reverse edge exists flag
    "Degree_Product",  # [9] src_deg × tgt_deg
    "Degree_Ratio",  # [10] src_deg / tgt_deg
]


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


def graph_node_attack_type(g, node_idx: int, default: int | None = None) -> int | None:
    """Get per-node attack_type for a single node, or *default* if unavailable."""
    if hasattr(g, "node_attack_type") and g.node_attack_type is not None:
        return int(g.node_attack_type[node_idx].item())
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
# Stage → model type mapping (single source of truth)
# ---------------------------------------------------------------------------
STAGE_MODEL_MAP: dict[str, str] = {
    "autoencoder": "vgae",
    "curriculum": "gat",
    "normal": "gat",
    "fusion": "dqn",
}

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
DEFAULT_SEEDS: list[int] = [42, 123, 456, 789, 1024]
