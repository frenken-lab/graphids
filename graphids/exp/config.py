"""Typed experiment config objects.

This is intentionally framework-agnostic: CLI, SLURM, and future adapters
consume the same domain config.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from graphids.core.data.preprocessing.representations import (
    RepresentationCfg,
    TemporalRepresentationCfg,
    representation_kind,
)
from graphids.exp.journal import RunManifest

Stage = Literal["fit", "test"]


def _representation_payload(cfg: RepresentationCfg) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    raise TypeError(f"unsupported representation config: {type(cfg)!r}")


def _json_payload(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    elif hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(k): _json_payload(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_json_payload(v) for v in value]
    return value


def _find_data_representation_payload(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump") and not isinstance(value, Mapping):
        value = value.model_dump(mode="json")
    if not isinstance(value, Mapping):
        return None
    if "representation_cfg" in value:
        payload = _json_payload(value["representation_cfg"])
        if not isinstance(payload, dict):
            raise TypeError("data representation_cfg must resolve to a mapping")
        return payload
    for nested in value.values():
        found = _find_data_representation_payload(nested)
        if found is not None:
            return found
    return None


def _payload(cfg: Any) -> dict[str, Any]:
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"unsupported run payload: {type(cfg)!r}")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResourceConfig(_StrictModel):
    """Execution resources for a run."""

    cluster: str | None = None
    partition: str | None = None
    accelerator: Literal["cpu", "gpu"] = "cpu"
    cpus_per_worker: int = 1
    gpus_per_worker: float = 0.0
    memory_gb: int | None = None
    time_limit: str | None = None
    gres: str | None = None
    account: str | None = None


class FitRunPayload(_StrictModel):
    """Typed payload for ``fit``/``test`` launches."""

    model: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    trainer: dict[str, Any] = Field(default_factory=dict)
    callbacks: dict[str, Any] = Field(default_factory=dict)
    loss_fn: dict[str, Any] | None = None
    seed_everything: int | None = None
    ckpt_path: str | None = None


RunPayload = FitRunPayload


class OutputConfig(_StrictModel):
    """Filesystem layout for a run."""

    run_dir: Path
    manifest_name: str = "manifest.json"
    events_name: str = "events.jsonl"
    artifact_dir: str = "artifacts"
    checkpoint_dir: str = "checkpoints"

    def journal_dir(self) -> Path:
        return self.run_dir / ".graphids"

    def manifest_path(self) -> Path:
        return self.journal_dir() / self.manifest_name

    def events_path(self) -> Path:
        return self.journal_dir() / self.events_name

    def artifact_path(self) -> Path:
        return self.run_dir / self.artifact_dir

    def checkpoint_path(self) -> Path:
        return self.run_dir / self.checkpoint_dir


class RunConfig(_StrictModel):
    """Resolved launch config for a single run."""

    name: str
    stage: Stage
    dataset: str | None = None
    seed: int = 42
    plan_id: str | None = None
    git_sha: str = "unknown"
    representation_cfg: RepresentationCfg = Field(default_factory=TemporalRepresentationCfg)
    payload: RunPayload = Field(default_factory=FitRunPayload)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    outputs: OutputConfig

    def mlflow_tags(self) -> dict[str, str]:
        tags = {
            "graphids.stage": self.stage,
            "graphids.run_dir": str(self.outputs.run_dir),
            "graphids.git_sha": self.git_sha,
        }
        if self.dataset is not None:
            tags["graphids.dataset"] = self.dataset
        if self.seed is not None:
            tags["graphids.seed"] = str(self.seed)
        if self.plan_id is not None:
            tags["graphids.plan_id"] = self.plan_id
        tags["graphids.representation"] = representation_kind(self.representation_cfg)
        return tags

    def mlflow_hparams(self) -> dict[str, Any]:
        return {
            "graphids.stage": self.stage,
            "graphids.dataset": self.dataset or "",
            "graphids.seed": self.seed if self.seed is not None else -1,
            "graphids.plan_id": self.plan_id or "",
            "graphids.git_sha": self.git_sha,
            "graphids.run_dir": str(self.outputs.run_dir),
            "graphids.representation": representation_kind(self.representation_cfg),
            "graphids.representation_cfg": _representation_payload(self.representation_cfg),
            "graphids.payload": _payload(self.payload),
            **{
                f"graphids.resource.{k}": v
                for k, v in self.resources.model_dump(mode="json").items()
            },
        }

    def journal_manifest(self, *, status: str, failure: str | None = None) -> RunManifest:
        return RunManifest(
            run_id=self.name,
            name=self.name,
            stage=self.stage,
            git_sha=self.git_sha,
            run_dir=str(self.outputs.run_dir),
            config={
                "representation_cfg": _representation_payload(self.representation_cfg),
                "payload": _payload(self.payload),
                "resources": self.resources.model_dump(mode="json"),
            },
            outputs={
                "journal_dir": str(self.outputs.journal_dir()),
                "manifest": str(self.outputs.manifest_path()),
                "events": str(self.outputs.events_path()),
                "artifact_dir": str(self.outputs.artifact_path()),
                "checkpoint_dir": str(self.outputs.checkpoint_path()),
            },
            status=status,
            failure=failure,
        )


class ExperimentConfig(_StrictModel):
    """Top-level experiment description.

    This is the domain entrypoint: compose here, then resolve to one
    ``RunConfig``.
    """

    experiment_name: str
    dataset: str
    seed: int = 42
    git_sha: str = "unknown"
    plan_id: str | None = None
    representation_cfg: RepresentationCfg = Field(default_factory=TemporalRepresentationCfg)
    config: dict[str, Any] = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    stage: Stage = "fit"

    def _run_dir(self, name: str, *, output_suffix: str | None = None) -> Path:
        from graphids.paths import trial_dir

        run_dir = trial_dir() / self.dataset / self.experiment_name / name
        if output_suffix:
            run_dir = run_dir / output_suffix
        return run_dir

    def build_run(
        self,
        *,
        name: str,
        stage: Stage,
        config: dict[str, Any] | None = None,
        output_suffix: str | None = None,
    ) -> RunConfig:
        cfg = {**self.config, **(config or {})}
        payload: RunPayload
        data_representation = _find_data_representation_payload(cfg.get("data", {}))
        if data_representation is not None:
            run_representation = _json_payload(_representation_payload(self.representation_cfg))
            if data_representation != run_representation:
                raise ValueError(
                    "fit/test data.source.representation_cfg must match top-level "
                    "representation_cfg so run metadata and materialized data cannot drift"
                )
        payload = FitRunPayload.model_validate(
            {
                "model": cfg.get("model", {}),
                "data": cfg.get("data", {}),
                "trainer": cfg.get("trainer", {}),
                "callbacks": cfg.get("callbacks", {}),
                "loss_fn": cfg.get("loss_fn"),
                "seed_everything": cfg.get("seed_everything", self.seed),
                "ckpt_path": cfg.get("ckpt_path"),
            }
        )
        return RunConfig(
            name=name,
            stage=stage,
            dataset=self.dataset,
            seed=self.seed,
            plan_id=self.plan_id,
            git_sha=self.git_sha,
            representation_cfg=self.representation_cfg,
            payload=payload,
            resources=self.resources,
            outputs=OutputConfig(run_dir=self._run_dir(name, output_suffix=output_suffix)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
        if cfg.get("resources") is None:
            cfg["resources"] = {}
        if cfg.get("config") is None:
            cfg["config"] = {}
        return cls.model_validate(cfg)
