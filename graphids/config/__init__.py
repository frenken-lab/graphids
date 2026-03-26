"""Configuration layer: constants, schema, paths, resolution.

Config composition uses plain YAML loading, dataclass defaults, and recursive
dict merge. All public APIs return _Namespace.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
# Plain namespace: config objects returned by resolve()
# ---------------------------------------------------------------------------


class _Namespace(types.SimpleNamespace):
    """Recursive namespace with attribute access, .get(), and bracket access.

    Returned by resolve() and to_namespace(). Supports attribute access,
    .get(key, default), and bracket access.
    """

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


def _dict_to_ns(d):
    """Recursively convert a dict to a _Namespace."""
    if isinstance(d, dict):
        return _Namespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(v) for v in d]
    return d


def _ns_to_dict(obj):
    """Recursively convert a _Namespace to a plain dict."""
    if isinstance(obj, types.SimpleNamespace):
        return {k: _ns_to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {k: _ns_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_ns_to_dict(v) for v in obj)
    return obj


def to_namespace(cfg) -> _Namespace:
    """Convert any config representation to a plain _Namespace.

    Handles: _Namespace (no-op), dict (from checkpoints or plain dicts).
    """
    if isinstance(cfg, _Namespace):
        return cfg
    if isinstance(cfg, dict):
        return _dict_to_ns(cfg)
    raise TypeError(f"Cannot convert {type(cfg).__name__} to _Namespace")


# ---------------------------------------------------------------------------
# Identity hash: plain function
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
    lake_root: str = ""  # filled from KD_GAT_LAKE_ROOT env var in resolve()
    device: str = "cuda"
    num_workers: int = 4
    production: bool = False
    auxiliaries: list = field(default_factory=list)
    # Data-derived dimensions — populated by CANBusDataModule.populate_config()
    num_ids: int = 0  # CAN arbitration-ID vocabulary size (for nn.Embedding)
    in_channels: int = 0  # node feature dimension (CAN ID col + continuous features)
    num_classes: int = 2  # number of target classes (derived from data, default binary)
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
# Config resolution helpers
# ---------------------------------------------------------------------------


def _parse_dotlist(overrides: tuple[str, ...] | list[str]) -> dict:
    """Parse CLI overrides like 'training.lr=0.001' into a nested dict."""
    result: dict = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # yaml.safe_load handles bool, int, float, list, null, string
        parsed = yaml.safe_load(value)
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = parsed
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge *override* into *base* in-place. Override wins for leaves."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _compute_derived(cfg: _Namespace) -> None:
    """Fill fields that were previously YAML interpolations.

    Must run after all merges so identity hashes see final config values.
    """
    if not cfg.lake_root:
        cfg.lake_root = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
    env_prod = os.environ.get("KD_GAT_PRODUCTION")
    if env_prod is not None:
        cfg.production = env_prod.lower() == "true"

    user = os.environ.get("USER", "unknown")
    cfg._tier = f"dev/{user}"
    cfg._output_base = f"{cfg.lake_root}/{cfg._tier}/{cfg.dataset}"

    # Checkpoint paths: {base}/{model}_{scale}_{stage}{hash}/seed_{seed}/best_model.ckpt
    # Keyed by model_type; stage comes from STAGE_MODEL_MAP inverse.
    _CKPT_STAGES = {
        "vgae": "autoencoder",
        "gat": cfg.gat_stage,
        "dqn": "fusion",
        "dgi": "autoencoder",
        "temporal": "temporal",
    }
    # temporal uses "gat" as model dir prefix (spatial encoder is GAT)
    _CKPT_MODEL = {"temporal": "gat"}
    cfg.checkpoints = _Namespace(**{
        model: (
            f"{cfg._output_base}/{_CKPT_MODEL.get(model, model)}_{cfg.scale}_{stage}"
            f"{compute_identity_hash(stage, cfg)}/seed_{cfg.seed}/best_model.ckpt"
        )
        for model, stage in _CKPT_STAGES.items()
    })


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


def resolve(*overrides: str) -> _Namespace:
    """Compose config: dataclass defaults + env vars + model preset + CLI overrides.

    Merge order: dataclass defaults → env-derived defaults → model preset → CLI overrides.
    Returns a plain _Namespace with all derived fields (checkpoints, paths) computed.
    """
    # 1. Dataclass defaults
    cfg = dataclasses.asdict(Config())

    # 2. Env-derived defaults (replaces config.yaml oc.env interpolations)
    cfg["lake_root"] = os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")
    cfg["production"] = os.environ.get("KD_GAT_PRODUCTION", "false").lower() == "true"

    # 3. Parse CLI overrides
    cli = _parse_dotlist(overrides)

    # 4. Apply CLI to determine model_type + scale (needed for preset lookup)
    _deep_merge(cfg, cli)

    # 5. Load and apply model preset from presets/{model_type}_{scale}.yaml
    preset_key = f"{cfg['model_type']}_{cfg['scale']}"
    preset_path = CONFIG_DIR / "presets" / f"{preset_key}.yaml"
    if preset_path.exists():
        preset = yaml.safe_load(preset_path.read_text()) or {}
        if preset:
            _deep_merge(cfg, preset)
    elif cfg["model_type"] in VALID_MODEL_TYPES:
        import structlog
        structlog.get_logger().warning(
            "missing_model_preset", key=preset_key, using="dataclass defaults",
        )

    # 6. Re-apply CLI overrides (CLI always wins over preset)
    _deep_merge(cfg, cli)

    # 7. Convert to namespace and compute derived fields
    ns = _dict_to_ns(cfg)
    _compute_derived(ns)
    return ns
