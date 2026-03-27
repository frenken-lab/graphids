"""Config dataclasses — single source of truth for all default values."""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Callback toggles (control which Lightning callbacks are active)
    swa_enabled: bool = True
    swa_lrs: float = 0.001
    swa_epoch_start: float = 0.75
    device_stats: bool = True
    lr_monitor: bool = True
    lr_monitor_interval: str = "step"


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
