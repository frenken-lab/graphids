"""Configuration layer: constants, schema, paths, resolution.

Config resolution uses jsonargparse for CLI parsing, type coercion, and
YAML preset loading. All public APIs return jsonargparse.Namespace.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from jsonargparse import ArgumentParser, Namespace

from .constants import (  # noqa: F401
    CATALOG_PATH,
    CONFIG_DIR,
    EXCLUDED_ATTACK_TYPES,
    MAX_DATA_BYTES,
    PREPROCESSING_VERSION,
    PROJECT_ROOT,
    STAGE_DEPENDENCIES,
    STAGE_MODEL_MAP,
    STAGES,
    PIPELINE_YAML,
    VALID_MODEL_TYPES,
    VALID_SCALES,
    compute_preprocessing_hash,
)


# ---------------------------------------------------------------------------
# Namespace conversion (checkpoint reload path)
# ---------------------------------------------------------------------------


def to_namespace(cfg):
    """Convert dict (from checkpoint reload) to jsonargparse Namespace.

    Recursively converts nested dicts and list items. No-op on Namespace.
    """
    if isinstance(cfg, Namespace):
        return cfg
    if isinstance(cfg, dict):
        return Namespace(**{k: to_namespace(v) for k, v in cfg.items()})
    if isinstance(cfg, list):
        return [to_namespace(v) for v in cfg]
    return cfg


# ---------------------------------------------------------------------------
# Identity hash
# ---------------------------------------------------------------------------


def compute_identity_hash(stage: str, cfg) -> str:
    """Compute identity hash for a stage from its identity_keys.

    Returns ``"_<8-char-hex>"`` or ``""`` if the stage has no identity keys.
    """
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
# Config schema (dataclasses with defaults)
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
    variational: bool = True  # False = GAE (no KL, no reparameterization)
    mask_ratio: float = 0.3
    canid_weight: float = 0.1
    nbr_weight: float = 0.05
    kl_weight: float = 0.01
    k_neg: int = 32  # negative samples per node for neighborhood loss


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
class DGIConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [480, 240, 48])
    latent_dim: int = 48
    heads: int = 4
    embedding_dim: int = 32
    dropout: float = 0.15
    conv_type: str = "gatv2"
    edge_dim: int = 11
    proj_dim: int = 0


@dataclass
class TrainingConfig:
    lr: float = 0.003
    max_epochs: int = 300
    batch_size: int = 8192
    patience: int = 100
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    precision: str = "16-mixed"
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
    loss_landscape: bool = False
    landscape_resolution: int = 51
    landscape_scale: float = 1.0


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
    gat_stage: str = "curriculum"  # which GAT training stage to use (curriculum or normal)
    seed: int = 42
    lake_root: str = "experimentruns"  # overridden by KD_GAT_LAKE_ROOT env var via env_prefix
    device: str = "cuda"
    num_workers: int = 4
    production: bool = False
    auxiliaries: list = field(default_factory=list)
    # Data-derived dimensions — populated by CANBusDataModule.populate_config()
    num_ids: int = 0
    in_channels: int = 0
    num_classes: int = 2
    vgae: VGAEConfig = field(default_factory=VGAEConfig)
    gat: GATConfig = field(default_factory=GATConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    dgi: DGIConfig = field(default_factory=DGIConfig)
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
# Derived fields (computed after parsing)
# ---------------------------------------------------------------------------


def _compute_derived(cfg: Namespace) -> None:
    """Fill path fields after all overrides are applied."""
    user = os.environ.get("USER", "unknown")
    cfg._tier = f"dev/{user}"
    cfg._output_base = f"{cfg.lake_root}/{cfg._tier}/{cfg.dataset}"

    _CKPT_STAGES = {
        "vgae": "autoencoder",
        "gat": cfg.gat_stage,
        "dqn": "fusion",
        "dgi": "autoencoder",
        "temporal": "temporal",
    }
    _CKPT_MODEL = {"temporal": "gat"}
    cfg.checkpoints = Namespace(**{
        model: (
            f"{cfg._output_base}/{_CKPT_MODEL.get(model, model)}_{cfg.scale}_{stage}"
            f"{compute_identity_hash(stage, cfg)}/seed_{cfg.seed}/best_model.ckpt"
        )
        for model, stage in _CKPT_STAGES.items()
    })


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve(*overrides: str) -> Namespace:
    """Compose config: dataclass defaults + YAML preset + CLI overrides.

    jsonargparse handles type coercion, nested dataclass flattening, and
    YAML/CLI merge. Preset file selected from model_type + scale.
    """
    # Extract model_type + scale from top-level overrides for preset lookup
    top = {}
    for o in overrides:
        if "=" in o and "." not in o:
            k, v = o.split("=", 1)
            top[k] = v

    preset_path = CONFIG_DIR / "presets" / f"{top.get('model_type', 'vgae')}_{top.get('scale', 'large')}.yaml"
    defaults = [str(preset_path)] if preset_path.exists() else []

    parser = ArgumentParser(default_config_files=defaults, env_prefix="KD_GAT", default_env=True)
    parser.add_class_arguments(Config, nested_key=None)

    args = [f"--{o}" if "=" in o and not o.startswith("-") else o for o in overrides]
    cfg = parser.parse_args(args)

    _compute_derived(cfg)
    return cfg
