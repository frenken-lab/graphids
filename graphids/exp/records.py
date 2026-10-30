"""Run manifests and event journals.

The goal is to make failure visible without needing Slurm, MLflow, or stderr as
the only source of truth.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from graphids._fs import atomic_write_json
from graphids.exp.config import RunConfig


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventRecord(_StrictModel):
    ts: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    status: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RunManifest(_StrictModel):
    # Version 1 manifests and their separate MLflow payloads remain readable.
    schema_version: Literal[1, 2] = 1
    run_id: str
    name: str
    stage: str
    git_sha: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    run_dir: str
    config: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    status: str = "created"
    failure: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    tracking_mode: Literal["offline", "online", "disabled"] = "offline"


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_dir: str
    status: str
    stage: str
    name: str
    last_event: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def write_config(run: RunConfig) -> Path:
    """Save the resolved input once; mutable execution state lives in the manifest."""
    path = run.outputs.run_dir / "resolved-run.yml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(run.model_dump(mode="json"), sort_keys=False))
    return path


def load_config(run_dir: str | Path) -> RunConfig:
    return RunConfig.model_validate(
        yaml.safe_load((Path(run_dir) / "resolved-run.yml").read_text())
    )


def make_manifest(run: RunConfig) -> RunManifest:
    """Keep historical manifest metadata while storing results in one place."""
    return RunManifest(
        schema_version=2,
        run_id=run.name,
        name=run.name,
        stage=run.stage,
        git_sha=run.git_sha,
        run_dir=str(run.outputs.run_dir),
        config=dict(
            representation_cfg=asdict(run.representation_cfg),
            payload=run.payload.model_dump(mode="json"),
            resources=run.resources.model_dump(mode="json"),
        ),
        outputs=dict(
            journal_dir=str(run.outputs.journal_dir()),
            manifest=str(run.outputs.manifest_path()),
            events=str(run.outputs.events_path()),
            artifact_dir=str(run.outputs.artifact_path()),
            checkpoint_dir=str(run.outputs.checkpoint_path()),
        ),
    )


def record_status(
    run: RunConfig,
    status: str,
    *,
    result: Mapping[str, Any] | None = None,
    failure: str | None = None,
    message: str | None = None,
    tracking_mode: str | None = None,
) -> RunManifest:
    """Publish authoritative status/results atomically, then append a history event."""
    manifest = load_manifest(run.outputs.run_dir, name=run.outputs.manifest_name)
    if manifest is None:
        manifest = make_manifest(run)
    changes = dict(schema_version=2, status=status, failure=failure)
    if tracking_mode is not None:
        changes["tracking_mode"] = tracking_mode
    if result is not None:
        changes.update(
            metrics=dict(result.get("metrics", {})),
            result={key: value for key, value in result.items() if key != "metrics"},
        )
    manifest = manifest.model_copy(update=changes)
    write_manifest(run.outputs.run_dir, manifest, name=run.outputs.manifest_name)
    append_event(
        run.outputs.run_dir,
        EventRecord(
            status=status,
            stage=run.stage,
            message=message or f"run_{status}",
            details={"failure": failure} if failure else {},
        ),
        name=run.outputs.events_name,
    )
    return manifest


def summarize_run(run_dir: str | Path) -> RunSummary | None:
    manifest = load_manifest(run_dir)
    if manifest is None:
        return None
    events = load_events(run_dir)
    return RunSummary(
        run_dir=manifest.run_dir,
        status=manifest.status,
        stage=manifest.stage,
        name=manifest.name,
        last_event=events[-1].message if events else None,
        error=manifest.failure,
        extra={"git_sha": manifest.git_sha, "run_id": manifest.run_id},
    )


def journal_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / ".graphids"


def manifest_path(run_dir: str | Path, *, name: str = "manifest.json") -> Path:
    return journal_dir(run_dir) / name


def events_path(run_dir: str | Path, *, name: str = "events.jsonl") -> Path:
    return journal_dir(run_dir) / name


def write_manifest(
    run_dir: str | Path, manifest: RunManifest, *, name: str = "manifest.json"
) -> Path:
    path = manifest_path(run_dir, name=name)
    atomic_write_json(path, manifest.model_dump(mode="json"))
    return path


def append_event(
    run_dir: str | Path, event: EventRecord, *, name: str = "events.jsonl"
) -> Path:
    path = events_path(run_dir, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(mode="json"), sort_keys=True))
        f.write("\n")
    return path


def load_manifest(
    run_dir: str | Path, *, name: str | None = None
) -> RunManifest | None:
    name = name or _record_name(run_dir, "manifest_name", "manifest.json")
    path = manifest_path(run_dir, name=name)
    if not path.exists():
        return None
    return RunManifest.model_validate(json.loads(path.read_text()))


def load_events(run_dir: str | Path, *, name: str | None = None) -> list[EventRecord]:
    name = name or _record_name(run_dir, "events_name", "events.jsonl")
    path = events_path(run_dir, name=name)
    if not path.exists():
        return []
    return [
        EventRecord.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _record_name(run_dir: str | Path, field: str, default: str) -> str:
    """Read layout metadata without revalidating historical training settings."""
    config = Path(run_dir) / "resolved-run.yml"
    if not config.exists():
        return default
    return yaml.safe_load(config.read_text()).get("outputs", {}).get(field, default)
