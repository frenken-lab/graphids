"""Configuration layer: constants, schema, paths, resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import MISSING, OmegaConf

from .constants import (  # noqa: F401
    CATALOG_PATH,
    CONFIG_DIR,
    DEFAULT_DATASET,
    DEFAULT_LAKE_ROOT,
    DEFAULT_SEEDS,
    EDGE_FEATURE_COUNT,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    NODE_FEATURE_COUNT,
    PREPROCESSING_DEFAULTS,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    SWEEP_RESULTS_DIR,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    compute_preprocessing_hash,
)

# ---------------------------------------------------------------------------
# Environment (KD_GAT_* env vars with defaults)
# ---------------------------------------------------------------------------

SLURM_ACCOUNT: str = os.environ.get("KD_GAT_SLURM_ACCOUNT", "PAS1266")
SLURM_PARTITION: str = os.environ.get("KD_GAT_SLURM_PARTITION", "gpu")
SLURM_GPU_TYPE: str = os.environ.get("KD_GAT_GPU_TYPE", "v100")
SWEEP_ID: str = os.environ.get("KD_GAT_SWEEP_ID", "")
USER_TAGS: str = os.environ.get("KD_GAT_TAGS", "")
CKPT_PATH: str = os.environ.get("KD_GAT_CKPT_PATH", "")


# ---------------------------------------------------------------------------
# Structured config schema (Hydra validates types via these dataclasses)
# ---------------------------------------------------------------------------

@dataclass
class VGAEConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [480, 240, 48])
    latent_dim: int = 48
    heads: int = 4
    embedding_dim: int = 32
    dropout: float = 0.15
    conv_type: str = "gatv2"
    edge_dim: int = 11
    proj_dim: int = 0
    mask_ratio: float = 0.3
    canid_weight: float = 0.1
    nbr_weight: float = 0.05
    kl_weight: float = 0.01


@dataclass
class GATConfig:
    hidden: int = 48
    layers: int = 3
    heads: int = 8
    dropout: float = 0.2
    embedding_dim: int = 16
    fc_layers: int = 3
    conv_type: str = "gatv2"
    edge_dim: int = 11
    pool_aggrs: list[str] = field(default_factory=lambda: ["mean"])
    proj_dim: int = 0


@dataclass
class DQNConfig:
    hidden: int = 576
    layers: int = 3
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01
    buffer_size: int = 100_000
    batch_size: int = 128
    target_update: int = 100
    weight_decay: float = 0.00001
    scheduler_patience: int = 1000
    max_patience: int = 5000
    vgae_error_weights: list[float] = field(default_factory=lambda: [0.4, 0.35, 0.25])
    reward_correct: float = 3.0
    reward_incorrect: float = -3.0
    confidence_weight: float = 0.5
    combined_conf_weight: float = 0.3
    disagreement_penalty: float = -1.0
    overconf_penalty: float = -1.5
    balance_weight: float = 0.3


@dataclass
class BanditConfig:
    ucb_alpha: float = 1.0
    lambda_reg: float = 1.0
    backbone_retrain_freq: int = 50
    backbone_lr: float = 0.001
    backbone_epochs: int = 5
    hidden: int = 576
    layers: int = 3
    buffer_size: int = 100_000
    batch_size: int = 128


@dataclass
class TrainingConfig:
    lr: float = 0.003
    max_epochs: int = 300
    batch_size: int = 4096
    patience: int = 100
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    precision: str = "16-mixed"
    safety_factor: float = 0.5
    gradient_checkpointing: bool = True
    use_teacher_cache: bool = True
    clear_cache_every_n: int = 100
    offload_teacher_to_cpu: bool = False
    accumulate_grad_batches: int = 1
    save_top_k: int = 1
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    log_every_n_steps: int = 50
    test_every_n_epochs: int = 5
    deterministic: bool = False
    cudnn_benchmark: bool = True
    compile_model: bool = False
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    scheduler_t_max: int = -1
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.1
    scheduler: str | None = None
    curriculum_start_ratio: float = 1.0
    curriculum_end_ratio: float = 10.0
    difficulty_percentile: float = 75.0
    use_vgae_mining: bool = True
    difficulty_cache_update: int = 10
    curriculum_memory_multiplier: float = 1.0
    log_teacher_student_comparison: bool = True
    dynamic_batching: bool = True
    loss_fn: str = "ce"  # ce | weighted_ce | focal
    loss_weight: float = 10.0  # weight for minority (attack) class in weighted_ce
    focal_gamma: float = 2.0  # focusing parameter for focal loss


@dataclass
class FusionConfig:
    method: str = "bandit"
    episodes: int = 500
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    episode_sample_size: int = 20_000
    training_step_interval: int = 32
    gpu_training_steps: int = 16
    lr: float = 0.001
    alpha_steps: int = 21
    mlp_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    mlp_max_epochs: int = 100
    decision_threshold: float = 0.5


@dataclass
class EvaluationConfig:
    batch_size: int = 256
    attention_sample_limit: int = 50
    cka_max_samples: int = 500


@dataclass
class TemporalConfig:
    enabled: bool = False
    temporal_window: int = 8
    temporal_stride: int = 1
    temporal_hidden: int = 64
    temporal_heads: int = 4
    temporal_layers: int = 2
    freeze_spatial: bool = True
    spatial_lr_factor: float = 0.1
    train_split: float = 0.8
    batch_size: int = 0  # 0 means use heuristic


@dataclass
class PreprocessingConfig:
    window_size: int = 100
    stride: int = 100
    train_val_split: float = 0.8


@dataclass
class Config:
    dataset: str = "hcrl_sa"
    model_type: str = "vgae"
    scale: str = "large"
    stage: str = "autoencoder"
    seed: int = 42
    lake_root: str = MISSING  # resolved from env/YAML
    device: str = "cuda"
    num_workers: int = 2
    production: bool = False
    auxiliaries: list = field(default_factory=list)
    # Data-derived dimensions — populated by CANBusDataModule.populate_config()
    num_ids: int = 0  # CAN arbitration-ID vocabulary size (for nn.Embedding)
    in_channels: int = 0  # node feature dimension (CAN ID col + continuous features)
    num_classes: int = 2  # number of target classes (derived from data, default binary)
    # Interpolation-heavy fields from config.yaml (Hydra needs them in schema)
    _tier: str = MISSING
    _output_base: str = MISSING
    checkpoints: Any = MISSING
    callbacks: Any = MISSING
    vgae: VGAEConfig = field(default_factory=VGAEConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


# ---------------------------------------------------------------------------
# Paths
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
# Config resolution
# ---------------------------------------------------------------------------

def resolve(*overrides: str):
    """Compose config: structured defaults + YAML infrastructure + model preset + CLI overrides.

    Merge order: dataclass defaults → config.yaml (infra) → model preset → CLI overrides.
    Hydra validates types via the structured Config dataclass.
    """
    schema = OmegaConf.structured(Config)
    infra = OmegaConf.load(CONFIG_DIR / "config.yaml")
    models = OmegaConf.load(CONFIG_DIR / "models.yaml")
    cli = OmegaConf.from_dotlist(list(overrides))

    # Open struct so infra keys (_tier, checkpoints, callbacks, hydra) can merge in
    OmegaConf.set_struct(schema, False)

    # Determine model_type + scale from overrides (before full merge)
    identity = OmegaConf.merge(schema, infra, cli)
    preset = models.get(f"{identity.model_type}_{identity.scale}") or {}

    cfg = OmegaConf.merge(schema, infra, preset, cli)
    OmegaConf.set_struct(cfg, True)
    return cfg
