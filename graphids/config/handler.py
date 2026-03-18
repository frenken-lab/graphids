"""Unified config handler: resolution, path derivation, constants.

All configuration state flows through ConfigHandler. External code imports from
graphids.config (which re-exports from this module). Artifact I/O lives in
graphids.pipeline.artifacts (not here — config is inert).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from .schema import PipelineConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File locations (derived from __file__)
# ---------------------------------------------------------------------------
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Standalone YAML loader (used by schema.py validators + ConfigHandler)
# ---------------------------------------------------------------------------
_pipeline_cache: dict | None = None


def load_pipeline_yaml() -> dict:
    """Load and cache pipeline.yaml."""
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text())
    return _pipeline_cache


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------------
# Environment settings (pydantic-settings: env vars with YAML fallbacks)
# ---------------------------------------------------------------------------


class EnvironmentSettings(BaseSettings):
    """Env var overrides for infrastructure defaults.

    All KD_GAT_* env vars are declared here. YAML values from pipeline.yaml
    and resources.yaml are passed as init defaults; env vars override them.
    """

    model_config = SettingsConfigDict(env_prefix="KD_GAT_")

    slurm_account: str = "PAS1266"
    slurm_partition: str = "gpu"
    gpu_type: str = "v100"
    experiment_root: str = "experimentruns"
    lake_root: str | None = None
    data_root: str | None = None
    cache_root: str | None = None
    stage_dir: str | None = None
    production: bool = False

    # MLFLOW_TRACKING_URI doesn't have KD_GAT_ prefix — handled separately
    mlflow_tracking_uri: str | None = Field(None, validation_alias="MLFLOW_TRACKING_URI")


class ConfigHandler:
    """Single API for config loading, resolution, and path derivation.

    Instantiated once as a module-level singleton in __init__.py. All state is
    eagerly loaded from YAML at init time; env vars are read via pydantic-settings.
    """

    def __init__(self) -> None:
        pipeline = load_pipeline_yaml()
        resources = yaml.safe_load((CONFIG_DIR / "resources.yaml").read_text())

        # --- Environment (pydantic-settings: env vars override YAML defaults) ---
        slurm = resources["slurm_defaults"]
        paths = pipeline["paths"]
        self.env = EnvironmentSettings(
            slurm_account=slurm["account"],
            slurm_partition=slurm["partition"],
            gpu_type=slurm["gpu_type"],
            experiment_root=paths["experiment_root"],
        )

        # --- Preprocessing constants ---
        prep = pipeline["preprocessing"]
        self.PREPROCESSING_VERSION: str = prep["version"]
        self.MAX_DATA_BYTES: int = prep["max_data_bytes"]
        self.NODE_FEATURE_COUNT: int = prep["node_feature_count"]
        self.EDGE_FEATURE_COUNT: int = prep["edge_feature_count"]
        self.EXCLUDED_ATTACK_TYPES: list[str] = prep["excluded_attack_types"]
        self.MMAP_TENSOR_LIMIT: int = prep["mmap_tensor_limit"]

        # --- Defaults ---
        defaults = pipeline["defaults"]
        self.DEFAULT_DATASET: str = defaults["dataset"]
        self.DEFAULT_SEEDS: list[int] = defaults["seeds"]

        # --- Path defaults ---
        self.SWEEP_RESULTS_DIR: str = paths["sweep_results_dir"]

        # --- Expose env settings as top-level attrs ---
        self.SLURM_ACCOUNT: str = self.env.slurm_account
        self.SLURM_PARTITION: str = self.env.slurm_partition
        self.SLURM_GPU_TYPE: str = self.env.gpu_type
        self.EXPERIMENT_ROOT: str = self.env.experiment_root
        self.MLFLOW_TRACKING_URI: str = (
            self.env.mlflow_tracking_uri
            or f"sqlite:///{PROJECT_ROOT / 'data' / 'mlflow' / 'mlflow.db'}"
        )

        # --- Filesystem ---
        self.PROJECT_ROOT: Path = PROJECT_ROOT
        self.CATALOG_PATH: Path = CONFIG_DIR / "datasets.yaml"

        # --- Pipeline topology ---
        self.STAGES: dict[str, tuple[str, str, str]] = {
            name: (s["learning_type"], s["model"], s["mode"])
            for name, s in pipeline["stages"].items()
        }
        self.STAGE_MODEL_MAP: dict[str, str] = {k: v[1] for k, v in self.STAGES.items()}
        self.VALID_MODEL_TYPES: frozenset[str] = frozenset(pipeline["models"].keys())
        self.VALID_SCALES: frozenset[str] = frozenset(pipeline["scales"])

        deps: dict[str, list[tuple[str, str]]] = {}
        for name, s in pipeline["stages"].items():
            dep_list = s.get("depends_on", [])
            if dep_list:
                deps[name] = [(d["model"], d["stage"]) for d in dep_list]
        self.STAGE_DEPENDENCIES: dict[str, list[tuple[str, str]]] = deps

        # --- Lazy state ---
        self._datasets: list[str] | None = None

    # ===================================================================
    # Resolution
    # ===================================================================

    def resolve(
        self,
        model_type: str,
        scale: str,
        auxiliaries: str = "none",
        **cli_overrides,
    ) -> PipelineConfig:
        """Compose config from YAML layers + CLI overrides → frozen PipelineConfig."""
        from .schema import PipelineConfig

        merged: dict = {}

        model_path = CONFIG_DIR / "models" / model_type / f"{scale}.yaml"
        if model_path.exists():
            _deep_merge(merged, yaml.safe_load(model_path.read_text()))

        if auxiliaries != "none":
            aux_path = CONFIG_DIR / "auxiliaries" / f"{auxiliaries}.yaml"
            if aux_path.exists():
                _deep_merge(merged, yaml.safe_load(aux_path.read_text()))

        if cli_overrides:
            _deep_merge(merged, cli_overrides)

        merged["model_type"] = model_type
        merged["scale"] = scale

        if self.env.experiment_root != "experimentruns":
            merged["experiment_root"] = self.env.experiment_root

        return PipelineConfig.model_validate(merged)

    def list_models(self) -> dict[str, list[str]]:
        """Discover available model types and scales from YAML files."""
        models = {}
        models_dir = CONFIG_DIR / "models"
        if models_dir.exists():
            for model_dir in sorted(models_dir.iterdir()):
                if model_dir.is_dir() and model_dir.name in self.VALID_MODEL_TYPES:
                    scales = [f.stem for f in sorted(model_dir.glob("*.yaml"))]
                    if scales:
                        models[model_dir.name] = scales
        return models

    def list_auxiliaries(self) -> list[str]:
        """Discover available auxiliary configs from YAML files."""
        aux_dir = CONFIG_DIR / "auxiliaries"
        if aux_dir.exists():
            return [f.stem for f in sorted(aux_dir.glob("*.yaml"))]
        return []

    # ===================================================================
    # Run identity
    # ===================================================================

    def run_id(self, cfg: PipelineConfig, stage: str) -> str:
        """Deterministic run ID: {dataset}/{model}_{scale}_{stage}[_{aux}]."""
        model, scale, aux_suffix = self._id_parts(cfg, stage)
        return f"{cfg.dataset}/{model}_{scale}_{stage}{aux_suffix}"

    @staticmethod
    def run_id_str(dataset: str, model_type: str, scale: str, stage: str, aux: str = "") -> str:
        """Run ID from raw strings (no PipelineConfig needed)."""
        suffix = f"_{aux}" if aux else ""
        model = "eval" if stage == "evaluation" else model_type
        return f"{dataset}/{model}_{scale}_{stage}{suffix}"

    def run_metadata(self, cfg: PipelineConfig, stage: str) -> dict[str, str]:
        """MLflow tags for a run."""
        return {
            "dataset": cfg.dataset,
            "model_type": cfg.model_type,
            "scale": cfg.scale,
            "stage": stage,
            "auxiliaries": cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
            "seed": str(cfg.seed),
            "run_group": self.run_id(cfg, stage),
            "config_hash": self._config_hash(cfg),
        }

    # ===================================================================
    # Path derivation
    # ===================================================================

    def stage_dir(self, cfg: PipelineConfig, stage: str) -> Path:
        """Canonical output directory for a stage."""
        if self.env.stage_dir:
            return Path(self.env.stage_dir) / self.run_id(cfg, stage) / f"seed_{cfg.seed}"

        if self.env.lake_root:
            aux = cfg.auxiliaries[0].type if cfg.auxiliaries else ""
            return self._lake_run_dir(
                cfg.dataset, cfg.model_type, cfg.scale, stage, aux=aux, seed=cfg.seed
            )

        return Path(cfg.experiment_root) / self.run_id(cfg, stage) / f"seed_{cfg.seed}"

    def checkpoint_path(self, cfg: PipelineConfig, stage: str) -> Path:
        """Where the best model checkpoint is saved."""
        return self.stage_dir(cfg, stage) / "best_model.pt"

    def config_path(self, cfg: PipelineConfig, stage: str) -> Path:
        """Where the frozen config JSON is saved alongside the model."""
        return self.stage_dir(cfg, stage) / "config.json"

    def metrics_path(self, cfg: PipelineConfig, stage: str) -> Path:
        """Where the evaluation metrics JSON is saved."""
        return self.stage_dir(cfg, stage) / "metrics.json"

    def data_dir(self, cfg: PipelineConfig) -> Path:
        """Raw data directory for a dataset."""
        if self.env.data_root:
            candidate = Path(self.env.data_root) / "raw" / cfg.dataset
            if candidate.exists():
                return candidate
        if self.env.lake_root:
            candidate = Path(self.env.lake_root) / "raw" / cfg.dataset
            if candidate.exists():
                return candidate
        return Path("data") / "automotive" / cfg.dataset

    def cache_dir(self, cfg: PipelineConfig) -> Path:
        """Processed-graph cache directory."""
        if self.env.cache_root:
            return Path(self.env.cache_root) / cfg.dataset
        if self.env.lake_root:
            return (
                Path(self.env.lake_root) / "cache" / f"v{self.PREPROCESSING_VERSION}" / cfg.dataset
            )
        if self.env.data_root:
            return Path(self.env.data_root) / "cache" / cfg.dataset
        return Path("data") / "cache" / cfg.dataset

    def sweep_result_path(self, stage: str, dataset: str, scale: str) -> Path:
        """Path for a sweep's best-config YAML."""
        if self.env.lake_root:
            return Path(self.env.lake_root) / "sweeps" / dataset / f"{stage}_{scale}_best.yaml"
        return self.PROJECT_ROOT / self.SWEEP_RESULTS_DIR / f"{stage}_{dataset}_{scale}_best.yaml"

    def sweep_searcher_path(self, stage: str, dataset: str, scale: str) -> Path:
        """Path for a sweep's Optuna searcher state pickle."""
        if self.env.lake_root:
            return Path(self.env.lake_root) / "sweeps" / dataset / f"{stage}_{scale}_searcher.pkl"
        return (
            self.PROJECT_ROOT / self.SWEEP_RESULTS_DIR / f"{stage}_{dataset}_{scale}_searcher.pkl"
        )

    def get_datasets(self) -> list[str]:
        """Dataset names from datasets.yaml (cached)."""
        if self._datasets is None:
            self._datasets = list(self.load_catalog().keys())
        return self._datasets

    def load_catalog(self) -> dict:
        """Load and validate all dataset entries from datasets.yaml."""
        from .schema import DatasetEntry

        raw = yaml.safe_load(self.CATALOG_PATH.read_text())
        return {name: DatasetEntry.model_validate(entry) for name, entry in raw.items()}

    # ===================================================================
    # Utilities
    # ===================================================================

    @staticmethod
    def parse_seeds(value: str) -> list[int]:
        """Parse seeds: comma-separated ints.

        Raises ValueError on invalid input (callers like argparse can wrap this).
        """
        if value is None:
            return []
        try:
            return [int(s.strip()) for s in value.split(",")]
        except ValueError as e:
            raise ValueError(f"Invalid seeds value '{value}': {e}") from e

    # ===================================================================
    # Private helpers
    # ===================================================================

    @staticmethod
    def _id_parts(cfg: PipelineConfig, stage: str) -> tuple[str, str, str]:
        aux_suffix = f"_{cfg.auxiliaries[0].type}" if cfg.auxiliaries else ""
        model = "eval" if stage == "evaluation" else cfg.model_type
        return model, cfg.scale, aux_suffix

    @staticmethod
    def _config_hash(cfg: PipelineConfig) -> str:
        return hashlib.sha256(
            json.dumps(cfg.model_dump(), sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    def _lake_run_dir(
        self,
        dataset: str,
        model_type: str,
        scale: str,
        stage: str,
        aux: str = "",
        seed: int = 42,
    ) -> Path:
        import getpass

        tier = "production" if self.env.production else f"dev/{getpass.getuser()}"
        model = "eval" if stage == "evaluation" else model_type
        suffix = f"_{aux}" if aux else ""
        return (
            Path(self.env.lake_root)
            / tier
            / dataset
            / f"{model}_{scale}_{stage}{suffix}"
            / f"seed_{seed}"
        )
