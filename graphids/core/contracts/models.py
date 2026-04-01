"""Training contract model definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingSpec(BaseModel):
    """Canonical execution input shared by CLI and orchestrators."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    model_family: str
    scale: str
    dataset: str
    seed: int
    run_dir: str
    config_files: tuple[str, ...]
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    upstream_ckpt_paths: dict[str, str] = Field(default_factory=dict)
    upstream_model_families: dict[str, str] = Field(default_factory=dict)
    runtime_overrides: dict[str, Any] = Field(default_factory=dict)


class ContractEnvelope(BaseModel):
    """Versioned wrapper for serialized contract payloads."""

    model_config = ConfigDict(extra="forbid")

    contract: str
    version: int
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
