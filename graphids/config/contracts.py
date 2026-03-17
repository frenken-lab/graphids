"""Pydantic data contracts for pipeline stage artifacts.

Validates stage outputs beyond file existence — checks content schema,
feature dimensions, and preprocessing compatibility.

Usage:
    from graphids.config.contracts import TrainingArtifact, EvaluationArtifact

    # Validate after stage completion
    artifact = TrainingArtifact.from_stage_dir(stage_path)

    # Validate preprocessing cache
    cache = PreprocessingArtifact.from_cache_dir(cache_path)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, model_validator

from .constants import EDGE_FEATURE_COUNT, NODE_FEATURE_COUNT, PREPROCESSING_VERSION
from .schema import PreprocessingConfig

_PREP_DEFAULTS = PreprocessingConfig()


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
        errors = []
        if self.preprocessing_version != PREPROCESSING_VERSION:
            errors.append(f"version {self.preprocessing_version} != {PREPROCESSING_VERSION}")
        if self.node_feature_count != NODE_FEATURE_COUNT:
            errors.append(f"node_features {self.node_feature_count} != {NODE_FEATURE_COUNT}")
        if self.edge_feature_count != EDGE_FEATURE_COUNT:
            errors.append(f"edge_features {self.edge_feature_count} != {EDGE_FEATURE_COUNT}")
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
    """Content-addressable hash of preprocessing parameters.

    Changes when window_size, stride, feature counts, or version change.
    Used as a secondary cache validation signal.
    """
    components = [
        PREPROCESSING_VERSION,
        str(NODE_FEATURE_COUNT),
        str(EDGE_FEATURE_COUNT),
        str(_PREP_DEFAULTS.window_size),
        str(_PREP_DEFAULTS.stride),
    ]
    return hashlib.sha256("|".join(components).encode()).hexdigest()[:16]
