"""Pydantic v2 config schema. Replaces the flat frozen dataclass.

One frozen BaseModel per concern. Nested composition. Declarative validation.
JSON serialization via model_dump_json / model_validate_json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

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
    dataset: str = "hcrl_ch"
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

    # --- Pipeline DAG ---
    stages: list[str] = Field(
        default=["autoencoder", "curriculum", "fusion", "evaluation"],
    )
    variants: list[VariantConfig] = Field(
        default=[
            VariantConfig(name="large", scale="large", needs_teacher=False),
            VariantConfig(
                name="small_kd", scale="small", auxiliaries="kd_standard", needs_teacher=True
            ),
            VariantConfig(name="small_nokd", scale="small", needs_teacher=False),
        ]
    )

    # --- Infrastructure ---
    schema_version: str = "1.0.0"
    experiment_root: str = "experimentruns"
    device: str = "cuda"
    num_workers: int = 0
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

    # --- Cross-field validation ---
    @model_validator(mode="after")
    def _check_cross_field(self) -> PipelineConfig:
        if self.model_type not in ("vgae", "gat", "dqn"):
            raise ValueError(f"model_type must be vgae/gat/dqn, got '{self.model_type}'")
        if self.scale not in ("large", "small"):
            raise ValueError(f"scale must be large/small, got '{self.scale}'")
        return self

    # --- Serialization ---
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load config. Handles nested (new) and flat (legacy) JSON."""
        raw = json.loads(Path(path).read_text())
        if "vgae" in raw and isinstance(raw.get("vgae"), dict):
            return cls.model_validate(raw)
        return cls._from_legacy_flat(raw)

    @classmethod
    def _from_legacy_flat(cls, flat: dict) -> PipelineConfig:
        """Convert legacy flat config.json to nested format.

        Deprecated: kept only for backward compatibility with old experimentruns/
        config.json files. All new configs use nested format.
        """
        import warnings

        warnings.warn(
            "Loading legacy flat config format — migrate to nested format",
            DeprecationWarning,
            stacklevel=3,
        )
        nested: dict[str, Any] = {}
        vgae: dict[str, Any] = {}
        gat: dict[str, Any] = {}
        dqn_arch: dict[str, Any] = {}
        training: dict[str, Any] = {}
        fusion: dict[str, Any] = {}
        kd_fields: dict[str, Any] = {}

        # Training fields that map to TrainingConfig
        training_field_names = set(TrainingConfig.model_fields.keys())

        # Fusion fields (with fusion_ prefix stripped)
        fusion_field_names = set(FusionConfig.model_fields.keys())

        for k, v in flat.items():
            # VGAE architecture
            if k.startswith("vgae_"):
                vgae[k.removeprefix("vgae_")] = v
            # GAT architecture
            elif k.startswith("gat_"):
                gat[k.removeprefix("gat_")] = v
            # DQN architecture
            elif k.startswith("dqn_"):
                dqn_arch[k.removeprefix("dqn_")] = v
            # KD fields
            elif k.startswith("kd_"):
                kd_fields[k.removeprefix("kd_")] = v
            # Fusion fields (fusion_ prefix)
            elif k.startswith("fusion_"):
                suffix = k.removeprefix("fusion_")
                if suffix in fusion_field_names:
                    fusion[suffix] = v
                else:
                    # fusion_episodes -> episodes, fusion_max_samples -> max_samples, etc.
                    training[k] = v  # will be filtered later
            # Training-level fields
            elif k in training_field_names:
                training[k] = v
            # Identity / infrastructure fields
            elif k == "model_size":
                # Map legacy "teacher"/"student" to "large"/"small"
                scale_map = {"teacher": "large", "student": "small"}
                nested["scale"] = scale_map.get(v, v)
            elif k == "use_kd":
                pass  # handled below
            elif k == "teacher_path":
                pass  # handled below
            elif k == "modality":
                pass  # dropped (no longer in schema)
            elif k in {
                "dataset",
                "model_type",
                "scale",
                "seed",
                "experiment_root",
                "device",
                "num_workers",
                "mp_start_method",
                "run_test",
            }:
                nested[k] = v
            else:
                # Remaining training fields not caught above
                if k in training_field_names:
                    training[k] = v
                # Fusion fields without prefix (from old flat format)
                elif k in {
                    "episode_sample_size",
                    "training_step_interval",
                    "gpu_training_steps",
                    "max_val_samples",
                    "alpha_steps",
                }:
                    fusion[k] = v

        # Map old fusion_ prefix fields
        if "episodes" not in fusion and "fusion_episodes" in flat:
            fusion["episodes"] = flat["fusion_episodes"]
        if "max_samples" not in fusion and "fusion_max_samples" in flat:
            fusion["max_samples"] = flat["fusion_max_samples"]
        if "lr" not in fusion and "fusion_lr" in flat:
            fusion["lr"] = flat["fusion_lr"]

        def _migrate_fields(src: dict, dest: dict, fields: tuple[str, ...]) -> None:
            for f in fields:
                if f in src and f not in dest:
                    dest[f] = src[f]

        _migrate_fields(
            flat,
            training,
            (
                "curriculum_start_ratio",
                "curriculum_end_ratio",
                "difficulty_percentile",
                "use_vgae_mining",
                "difficulty_cache_update",
                "curriculum_memory_multiplier",
                "log_teacher_student_comparison",
            ),
        )
        _migrate_fields(
            flat,
            training,
            (
                "use_scheduler",
                "scheduler_type",
                "scheduler_t_max",
                "scheduler_step_size",
                "scheduler_gamma",
            ),
        )
        _migrate_fields(
            flat,
            training,
            (
                "gradient_checkpointing",
                "use_teacher_cache",
                "clear_cache_every_n",
                "offload_teacher_to_cpu",
                "safety_factor",
                "accumulate_grad_batches",
                "save_top_k",
                "monitor_metric",
                "monitor_mode",
                "log_every_n_steps",
                "test_every_n_epochs",
                "deterministic",
                "cudnn_benchmark",
            ),
        )
        _migrate_fields(
            flat,
            training,
            (
                "lr",
                "max_epochs",
                "batch_size",
                "patience",
                "weight_decay",
                "gradient_clip",
                "precision",
            ),
        )

        if vgae:
            nested["vgae"] = vgae
        if gat:
            nested["gat"] = gat
        if dqn_arch:
            nested["dqn"] = dqn_arch
        if training:
            nested["training"] = training
        if fusion:
            nested["fusion"] = fusion

        # KD → auxiliaries
        if flat.get("use_kd"):
            kd_aux = {"type": "kd"} | {
                k: kd_fields[k]
                for k in ("temperature", "alpha", "vgae_latent_weight", "vgae_recon_weight")
                if k in kd_fields
            }
            if flat.get("teacher_path"):
                kd_aux["model_path"] = flat["teacher_path"]
            nested["auxiliaries"] = [kd_aux]

        return cls.model_validate(nested)
