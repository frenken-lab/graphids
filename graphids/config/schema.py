"""Pydantic v2 config schema — every data model in the config layer.

One frozen BaseModel per concern. Nested composition. Declarative validation.
JSON serialization via model_dump_json / model_validate_json.

Includes: pipeline config, architecture sub-configs, dataset catalog entries,
and artifact validation contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class VGAEArchitecture(BaseModel, frozen=True):
    hidden_dims: tuple[int, ...] = (480, 240, 48)
    latent_dim: int = Field(48, ge=1)
    heads: int = Field(4, ge=1)
    embedding_dim: int = Field(32, ge=1)
    dropout: float = Field(0.15, ge=0, le=1)
    conv_type: Literal["gat", "gatv2", "transformer"] = "gat"
    edge_dim: int = Field(11, ge=1)
    proj_dim: int = Field(0, ge=0)


class GATArchitecture(BaseModel, frozen=True):
    hidden: int = Field(48, ge=1)
    layers: int = Field(3, ge=1)
    heads: int = Field(8, ge=1)
    dropout: float = Field(0.2, ge=0, le=1)
    embedding_dim: int = Field(16, ge=1)
    fc_layers: int = Field(3, ge=1)
    conv_type: Literal["gat", "gatv2", "transformer"] = "gat"
    edge_dim: int = Field(11, ge=1)
    pool_aggrs: tuple[str, ...] = ("mean",)
    proj_dim: int = Field(0, ge=0)


class DQNArchitecture(BaseModel, frozen=True):
    hidden: int = Field(576, ge=1)
    layers: int = Field(3, ge=1)
    gamma: float = Field(0.99, gt=0, le=1)
    epsilon: float = Field(0.1, ge=0, le=1)
    epsilon_decay: float = Field(0.995, gt=0, le=1)
    min_epsilon: float = Field(0.01, ge=0)
    buffer_size: int = Field(100_000, ge=1)
    batch_size: int = Field(128, ge=1)
    target_update: int = Field(100, ge=1)
    weight_decay: float = Field(1e-5, ge=0)
    scheduler_patience: int = Field(1000, ge=1)
    max_patience: int = Field(5000, ge=1)


class AuxiliaryConfig(BaseModel, frozen=True):
    """One auxiliary loss modifier (KD, PINN, etc.). Flat with defaults."""

    type: Literal["kd"] = "kd"  # Extend Literal as new auxiliaries are added
    model_path: str = ""  # Explicit override; empty = auto-resolve from teacher_scale
    teacher_scale: str = "large"  # Scale of teacher model (auto-resolved when model_path empty)
    alpha: float = Field(0.7, ge=0, le=1)
    # KD-specific (defaults are safe no-ops for non-KD types)
    temperature: float = Field(4.0, gt=0)
    vgae_latent_weight: float = Field(0.5, ge=0, le=1)
    vgae_recon_weight: float = Field(0.5, ge=0, le=1)


class TrainingConfig(BaseModel, frozen=True):
    lr: float = Field(0.003, gt=0)
    max_epochs: int = Field(300, ge=1)
    batch_size: int = Field(4096, ge=1)
    patience: int = Field(100, ge=1)
    weight_decay: float = Field(1e-4, ge=0)
    gradient_clip: float = Field(1.0, gt=0)
    precision: str = "16-mixed"
    safety_factor: float = Field(0.5, gt=0, le=1)
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
    # LR scheduling
    use_scheduler: bool = False
    scheduler_type: str = "cosine"
    scheduler_t_max: int = -1
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.1
    # Curriculum params (used when stage=curriculum)
    curriculum_start_ratio: float = 1.0
    curriculum_end_ratio: float = 10.0
    difficulty_percentile: float = 75.0
    use_vgae_mining: bool = True
    difficulty_cache_update: int = 10
    curriculum_memory_multiplier: float = 1.0
    log_teacher_student_comparison: bool = True
    dynamic_batching: bool = True


class FusionConfig(BaseModel, frozen=True):
    method: Literal["dqn", "mlp", "weighted_avg"] = "dqn"
    episodes: int = Field(500, ge=1)
    max_samples: int = Field(150_000, ge=1)
    max_val_samples: int = Field(30_000, ge=1)
    episode_sample_size: int = Field(20_000, ge=1)
    training_step_interval: int = Field(32, ge=1)
    gpu_training_steps: int = Field(16, ge=1)
    lr: float = Field(0.001, gt=0)
    alpha_steps: int = Field(21, ge=1)
    # MLP-specific
    mlp_hidden_dims: tuple[int, ...] = (64, 32)
    mlp_max_epochs: int = Field(100, ge=1)


class PreprocessingConfig(BaseModel, frozen=True):
    window_size: int = Field(100, ge=1)
    stride: int = Field(100, ge=1)
    train_val_split: float = Field(0.8, gt=0.0, lt=1.0)
    chunk_size: int = Field(5000, ge=100)
    ray_file_threshold: int = Field(4, ge=1)


class TemporalConfig(BaseModel, frozen=True):
    enabled: bool = False
    temporal_window: int = Field(8, ge=2, le=32)
    temporal_stride: int = Field(1, ge=1)
    temporal_hidden: int = Field(64, ge=1)
    temporal_heads: int = Field(4, ge=1)
    temporal_layers: int = Field(2, ge=1)
    freeze_spatial: bool = True
    spatial_lr_factor: float = Field(0.1, gt=0, le=1.0)


class TuneConfig(BaseModel, frozen=True):
    """ASHA scheduler defaults for Ray Tune HPO."""

    grace_period: int = Field(10, ge=1)
    reduction_factor: int = Field(3, ge=2)


class VariantConfig(BaseModel, frozen=True):
    """A pipeline variant (e.g. large teacher, small KD, small ablation)."""

    name: str
    scale: str = "large"
    auxiliaries: str = "none"
    needs_teacher: bool = False
    stages: list[str] = Field(
        default=["autoencoder", "curriculum", "fusion", "evaluation"],
    )


class PipelineConfig(BaseModel, frozen=True):
    """Every tunable parameter lives here. Nowhere else."""

    # --- Identity (the four concerns) ---
    dataset: str = "hcrl_sa"
    model_type: str = "vgae"
    scale: str = "large"
    seed: int = 42

    # --- Architecture (per model type) ---
    vgae: VGAEArchitecture = VGAEArchitecture()
    gat: GATArchitecture = GATArchitecture()
    dqn: DQNArchitecture = DQNArchitecture()

    # --- Training ---
    training: TrainingConfig = TrainingConfig()
    auxiliaries: list[AuxiliaryConfig] = []
    fusion: FusionConfig = FusionConfig()
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    temporal: TemporalConfig = TemporalConfig()
    tune: TuneConfig = TuneConfig()

    # --- Pipeline DAG (defaults from pipeline.yaml) ---
    stages: list[str] = Field(default=None)
    variants: list[VariantConfig] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def _fill_pipeline_defaults(cls, data):
        """Fill stages and variants from pipeline.yaml when not provided."""
        if isinstance(data, dict):
            from .handler import load_pipeline_yaml

            pipeline = load_pipeline_yaml()
            if data.get("stages") is None:
                data["stages"] = pipeline.get(
                    "default_stages", ["autoencoder", "curriculum", "fusion", "evaluation"]
                )
            if data.get("variants") is None:
                data["variants"] = [
                    {"name": name, **v} for name, v in pipeline.get("variants", {}).items()
                ]
        return data

    # --- Infrastructure ---
    experiment_root: str = "experimentruns"
    device: str = "cuda"
    num_workers: int = 2
    mp_start_method: str = "spawn"
    run_test: bool = True

    # --- Convenience properties ---
    @property
    def has_kd(self) -> bool:
        return any(a.type == "kd" for a in self.auxiliaries)

    @property
    def kd(self) -> AuxiliaryConfig | None:
        return next((a for a in self.auxiliaries if a.type == "kd"), None)

    @property
    def active_arch(self):
        """Return the architecture config for the active model_type."""
        return getattr(self, self.model_type)

    # --- Cross-field validation (reads valid values from pipeline.yaml) ---
    @model_validator(mode="after")
    def _check_cross_field(self) -> PipelineConfig:
        from .handler import load_pipeline_yaml

        _pl = load_pipeline_yaml()
        VALID_MODEL_TYPES = frozenset(_pl["models"].keys())
        VALID_SCALES = frozenset(_pl["scales"])

        if self.model_type not in VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(VALID_MODEL_TYPES)}, got '{self.model_type}'"
            )
        if self.scale not in VALID_SCALES:
            raise ValueError(f"scale must be one of {sorted(VALID_SCALES)}, got '{self.scale}'")
        return self

    # --- Serialization ---
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load config from JSON file."""
        raw = json.loads(Path(path).read_text())
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Dataset catalog entry
# ---------------------------------------------------------------------------


class DatasetEntry(BaseModel, frozen=True):
    """Schema for one dataset catalog entry in datasets.yaml."""

    domain: str
    protocol: str
    source: str = ""
    description: str = ""
    csv_dir: str
    csv_columns: dict[str, str]
    train_subdir: str
    train_attack_subdir: str = ""
    test_subdirs: list[str]
    added_by: str = ""


# ---------------------------------------------------------------------------
# Artifact validation contracts
# ---------------------------------------------------------------------------


class StageArtifact(BaseModel, frozen=True):
    """Base contract for pipeline stage outputs."""

    stage_dir: Path
    config_json: Path
    metrics_json: Path

    @model_validator(mode="after")
    def _validate_files_exist(self) -> StageArtifact:
        missing = []
        if not self.config_json.exists():
            missing.append(str(self.config_json))
        if not self.metrics_json.exists():
            missing.append(str(self.metrics_json))
        if missing:
            raise ValueError(f"Missing required files: {missing}")
        return self

    @classmethod
    def from_stage_dir(cls, path: Path) -> StageArtifact:
        return cls(
            stage_dir=path,
            config_json=path / "config.json",
            metrics_json=path / "metrics.json",
        )


class TrainingArtifact(StageArtifact, frozen=True):
    """Contract: training stages (autoencoder, curriculum, fusion)."""

    best_model_pt: Path = Path()

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> TrainingArtifact:
        if not self.best_model_pt.exists():
            raise ValueError(f"Missing checkpoint: {self.best_model_pt}")
        return self

    @classmethod
    def from_stage_dir(cls, path: Path) -> TrainingArtifact:
        return cls(
            stage_dir=path,
            config_json=path / "config.json",
            metrics_json=path / "metrics.json",
            best_model_pt=path / "best_model.pt",
        )


class EvaluationArtifact(StageArtifact, frozen=True):
    """Contract: evaluation stage."""

    embeddings_npz: Path | None = None
    dqn_policy_json: Path | None = None

    @classmethod
    def from_stage_dir(cls, path: Path) -> EvaluationArtifact:
        emb = path / "embeddings.npz"
        pol = path / "dqn_policy.json"
        return cls(
            stage_dir=path,
            config_json=path / "config.json",
            metrics_json=path / "metrics.json",
            embeddings_npz=emb if emb.exists() else None,
            dqn_policy_json=pol if pol.exists() else None,
        )


class PreprocessingArtifact(BaseModel, frozen=True):
    """Contract: preprocessing cache output."""

    cache_dir: Path
    metadata_json: Path
    preprocessing_version: str
    node_feature_count: int
    edge_feature_count: int
    config_hash: str = ""

    @model_validator(mode="after")
    def _validate_compatibility(self) -> PreprocessingArtifact:
        from .handler import load_pipeline_yaml

        prep = load_pipeline_yaml()["preprocessing"]
        errors = []
        if self.preprocessing_version != prep["version"]:
            errors.append(f"version {self.preprocessing_version} != {prep['version']}")
        if self.node_feature_count != prep["node_feature_count"]:
            errors.append(
                f"node_features {self.node_feature_count} != {prep['node_feature_count']}"
            )
        if self.edge_feature_count != prep["edge_feature_count"]:
            errors.append(
                f"edge_features {self.edge_feature_count} != {prep['edge_feature_count']}"
            )
        if errors:
            raise ValueError(f"Preprocessing cache incompatible: {'; '.join(errors)}")
        return self

    @classmethod
    def from_cache_dir(cls, path: Path) -> PreprocessingArtifact:
        metadata_path = path / "cache_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"No cache_metadata.json in {path}")

        metadata = json.loads(metadata_path.read_text())
        return cls(
            cache_dir=path,
            metadata_json=metadata_path,
            preprocessing_version=metadata.get("preprocessing_version", "unknown"),
            node_feature_count=metadata.get("node_feature_dim", 0),
            edge_feature_count=metadata.get("edge_feature_dim", 0),
            config_hash=metadata.get("config_hash", ""),
        )


def compute_preprocessing_hash() -> str:
    """Content-addressable hash of preprocessing parameters."""
    import hashlib

    from .handler import load_pipeline_yaml

    prep = load_pipeline_yaml()["preprocessing"]
    defaults = PreprocessingConfig()
    components = [
        prep["version"],
        str(prep["node_feature_count"]),
        str(prep["edge_feature_count"]),
        str(defaults.window_size),
        str(defaults.stride),
        str(defaults.train_val_split),
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
