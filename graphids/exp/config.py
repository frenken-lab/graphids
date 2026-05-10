"""Typed experiment config objects.

This is intentionally framework-agnostic so we can feed it from Hydra later or
instantiate it directly from CLI/YAML today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from graphids.exp.journal import RunManifest
from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResourceConfig(_StrictModel):
    """Execution resources for a run."""

    backend: Literal["local", "ray"] = "local"
    accelerator: Literal["cpu", "gpu"] = "cpu"
    num_workers: int = 0
    cpus_per_worker: int = 1
    gpus_per_worker: float = 0.0
    memory_gb: int | None = None


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

    def mlflow_dir(self) -> Path:
        return self.run_dir / ".mlflow"


class IdentityConfig(_StrictModel):
    run_name: str
    run_dir: str
    jobname: str


class MetaConfig(_StrictModel):
    group: str
    variant: str
    dataset: str
    seed: int
    model_type: str
    scale: str
    subdir: str = "ablations"


class UpstreamConfig(_StrictModel):
    role: str
    ckpt_path: str
    ckpt_tla: str


class RunConfig(_StrictModel):
    """Resolved launch config for a single run."""

    name: str
    stage: Literal["fit", "test", "extract", "analyze", "cache", "hf_push"]
    dataset: str | None = None
    seed: int = 42
    plan_id: str | None = None
    git_sha: str = "unknown"
    config: dict[str, Any] = Field(default_factory=dict)
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
        return tags

    def mlflow_hparams(self, *, backend: str) -> dict[str, Any]:
        return {
            "graphids.stage": self.stage,
            "graphids.dataset": self.dataset or "",
            "graphids.seed": self.seed if self.seed is not None else -1,
            "graphids.plan_id": self.plan_id or "",
            "graphids.git_sha": self.git_sha,
            "graphids.run_dir": str(self.outputs.run_dir),
            "graphids.backend": backend,
            **self.config,
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
            config={**self.config, "resources": self.resources.model_dump(mode="json")},
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

    This is the future Hydra entrypoint: compose here, then resolve to one or
    more RunConfig objects.
    """

    experiment_name: str
    dataset: str
    seed: int = 42
    git_sha: str = "unknown"
    plan_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    stage: Literal["fit", "test", "extract", "analyze", "cache", "hf_push"] = "fit"

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
        stage: Literal["fit", "test", "extract", "analyze", "cache", "hf_push"],
        config: dict[str, Any] | None = None,
        output_suffix: str | None = None,
    ) -> RunConfig:
        return RunConfig(
            name=name,
            stage=stage,
            dataset=self.dataset,
            seed=self.seed,
            plan_id=self.plan_id,
            git_sha=self.git_sha,
            config={**self.config, **(config or {})},
            resources=self.resources,
            outputs=OutputConfig(run_dir=self._run_dir(name, output_suffix=output_suffix)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        cfg = OmegaConf.load(path)
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))


class TrainConfig(_StrictModel):
    name: str
    action: Literal["fit", "test"]
    plan_id: str
    plan_module: str
    git_sha: str
    identity: IdentityConfig
    meta: MetaConfig
    model: Any
    loss_fn: Any | None = None
    data: Any
    trainer: dict[str, Any]
    callbacks: list[dict[str, Any]]
    seed_everything: int
    upstreams: list[UpstreamConfig] = Field(default_factory=list)
    resources: ResourceConfig


class ExtractConfig(_StrictModel):
    """Extract config for cached fusion dumps."""

    name: str
    action: Literal["extract"]
    plan_id: str
    dataset: str
    extractor_ckpts: dict[str, str]
    output_dir: str
    resources: ResourceConfig
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    batch_size: int = 256
    seed: int = 42
    val_fraction: float = 0.2
    representation_cfg: GraphRepresentationCfg = Field(default_factory=SnapshotRepresentationCfg)


class AnalyzeConfig(_StrictModel):
    """Artifact-analysis config for cached checkpoints and validation data."""

    name: str
    action: Literal["analyze"]
    plan_id: str
    resources: ResourceConfig

    ckpt_path: str
    dataset: str
    model_type: Literal["vgae", "dgi", "gat", "fusion"]
    output_dir: str
    lake_root: str

    embeddings: bool = True
    attention: bool = False
    cka: bool = False
    landscape: bool = False
    fusion_policy: bool = False

    cka_teacher_ckpt: str = ""
    cka_max_samples: int = 500

    landscape_resolution: int = 51
    landscape_scale: float = 1.0
    landscape_max_graphs: int = 500

    embedding_max_samples: int = 2000
    attention_max_samples: int = 50

    batch_size: int = 256
    seed: int = 42
    vocab_scope: str = "train"
    representation_cfg: GraphRepresentationCfg = Field(default_factory=SnapshotRepresentationCfg)

    vgae_ckpt_path: str = ""
    gat_ckpt_path: str = ""


class HFPushConfig(_StrictModel):
    name: str
    action: Literal["hf_push"]
    plan_id: str
    artifact_type: Literal["checkpoints", "cache", "states", "logs", "analysis"]
    repo_id: str
    repo_type: Literal["model", "dataset"] = "model"
    local_path: str
    path_in_repo: str
    private: bool = True
    resources: ResourceConfig


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Minimal status summary for UI/readout code."""

    run_dir: str
    status: str
    stage: str
    name: str
    last_event: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
