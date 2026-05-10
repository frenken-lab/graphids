"""Typed experiment config objects.

This is intentionally framework-agnostic so we can feed it from Hydra later or
instantiate it directly from CLI/YAML today.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from graphids.core.data.preprocessing.representations import (
    GraphRepresentationCfg,
    SnapshotRepresentationCfg,
    representation_kind,
)
from graphids.exp.journal import RunManifest


def _representation_payload(cfg: GraphRepresentationCfg) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "model_dump"):
        return cfg.model_dump(mode="json")
    raise TypeError(f"unsupported representation config: {type(cfg)!r}")


def _payload(cfg: RunPayload) -> dict[str, Any]:
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

    backend: Literal["local", "ray"] = "local"
    accelerator: Literal["cpu", "gpu"] = "cpu"
    num_workers: int = 0
    cpus_per_worker: int = 1
    gpus_per_worker: float = 0.0
    memory_gb: int | None = None


class FitRunPayload(_StrictModel):
    """Typed payload for ``fit``/``test`` launches."""

    model: dict[str, Any] = Field(default_factory=dict)
    data: dict[str, Any] = Field(default_factory=dict)
    trainer: dict[str, Any] = Field(default_factory=dict)
    callbacks: dict[str, Any] = Field(default_factory=dict)
    loss_fn: dict[str, Any] | None = None
    seed_everything: int | None = None
    ckpt_path: str | None = None


class ExtractRunPayload(_StrictModel):
    """Typed payload for ``extract`` launches."""

    dataset: str = ""
    output_dir: str = ""
    checkpoints: dict[str, str] | None = None
    extractor_ckpts: dict[str, str] | None = None
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    batch_size: int = 256
    seed: int | None = None
    val_fraction: float = 0.2


class AnalyzeRunPayload(_StrictModel):
    """Typed payload for ``analyze`` launches."""

    name: str = ""
    plan_id: str = ""
    ckpt_path: str = ""
    dataset: str = ""
    model_type: Literal["vgae", "dgi", "gat", "fusion"] = "gat"
    output_dir: str = ""
    lake_root: str = ""
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
    vgae_ckpt_path: str = ""
    gat_ckpt_path: str = ""


RunPayload = FitRunPayload | ExtractRunPayload | AnalyzeRunPayload


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


class RunConfig(_StrictModel):
    """Resolved launch config for a single run."""

    name: str
    stage: Literal["fit", "test", "extract", "analyze", "cache", "hf_push"]
    dataset: str | None = None
    seed: int = 42
    plan_id: str | None = None
    git_sha: str = "unknown"
    representation_cfg: GraphRepresentationCfg = Field(default_factory=SnapshotRepresentationCfg)
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

    def mlflow_hparams(self, *, backend: str) -> dict[str, Any]:
        return {
            "graphids.stage": self.stage,
            "graphids.dataset": self.dataset or "",
            "graphids.seed": self.seed if self.seed is not None else -1,
            "graphids.plan_id": self.plan_id or "",
            "graphids.git_sha": self.git_sha,
            "graphids.run_dir": str(self.outputs.run_dir),
            "graphids.backend": backend,
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

    This is the future Hydra entrypoint: compose here, then resolve to one or
    more RunConfig objects.
    """

    experiment_name: str
    dataset: str
    seed: int = 42
    git_sha: str = "unknown"
    plan_id: str | None = None
    representation_cfg: GraphRepresentationCfg = Field(default_factory=SnapshotRepresentationCfg)
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
        cfg = {**self.config, **(config or {})}
        payload: RunPayload
        if stage in {"fit", "test"}:
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
        elif stage == "extract":
            payload = ExtractRunPayload.model_validate(
                {
                    "dataset": cfg.get("dataset", self.dataset),
                    "output_dir": cfg.get("output_dir", ""),
                    "checkpoints": cfg.get("checkpoints"),
                    "extractor_ckpts": cfg.get("extractor_ckpts"),
                    "max_samples": cfg.get("max_samples", 150_000),
                    "max_val_samples": cfg.get("max_val_samples", 30_000),
                    "batch_size": cfg.get("batch_size", 256),
                    "seed": cfg.get("seed", self.seed),
                    "val_fraction": cfg.get("val_fraction", 0.2),
                }
            )
        elif stage == "analyze":
            payload = AnalyzeRunPayload.model_validate(
                {
                    "name": name,
                    "plan_id": self.plan_id or name,
                    "ckpt_path": cfg.get("ckpt_path", ""),
                    "dataset": cfg.get("dataset", self.dataset),
                    "model_type": cfg.get("model_type", "gat"),
                    "output_dir": cfg.get("output_dir", ""),
                    "lake_root": cfg.get("lake_root", ""),
                    "embeddings": cfg.get("embeddings", True),
                    "attention": cfg.get("attention", False),
                    "cka": cfg.get("cka", False),
                    "landscape": cfg.get("landscape", False),
                    "fusion_policy": cfg.get("fusion_policy", False),
                    "cka_teacher_ckpt": cfg.get("cka_teacher_ckpt", ""),
                    "cka_max_samples": cfg.get("cka_max_samples", 500),
                    "landscape_resolution": cfg.get("landscape_resolution", 51),
                    "landscape_scale": cfg.get("landscape_scale", 1.0),
                    "landscape_max_graphs": cfg.get("landscape_max_graphs", 500),
                    "embedding_max_samples": cfg.get("embedding_max_samples", 2000),
                    "attention_max_samples": cfg.get("attention_max_samples", 50),
                    "batch_size": cfg.get("batch_size", 256),
                    "seed": cfg.get("seed", self.seed),
                    "vocab_scope": cfg.get("vocab_scope", "train"),
                    "vgae_ckpt_path": cfg.get("vgae_ckpt_path", ""),
                    "gat_ckpt_path": cfg.get("gat_ckpt_path", ""),
                }
            )
        else:
            raise NotImplementedError(f"stage {stage!r} is not wired yet")
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
        cfg = OmegaConf.load(path)
        return cls.model_validate(OmegaConf.to_container(cfg, resolve=True))


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
