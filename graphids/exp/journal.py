"""Run manifests and event journals.

The goal is to make failure visible without needing Slurm, MLflow, or stderr as
the only source of truth.
"""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EventRecord(_StrictModel):
    ts: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    status: str
    stage: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RunManifest(_StrictModel):
    run_id: str
    name: str
    stage: str
    git_sha: str
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    run_dir: str
    config: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    status: str = "created"
    failure: str | None = None


def journal_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / ".graphids"


def manifest_path(run_dir: str | Path, *, name: str = "manifest.json") -> Path:
    return journal_dir(run_dir) / name


def events_path(run_dir: str | Path, *, name: str = "events.jsonl") -> Path:
    return journal_dir(run_dir) / name


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
        tmp = Path(f.name)
    tmp.replace(path)


def write_manifest(run_dir: str | Path, manifest: RunManifest, *, name: str = "manifest.json") -> Path:
    path = manifest_path(run_dir, name=name)
    _write_json_atomic(path, manifest.model_dump(mode="json"))
    return path


def append_event(run_dir: str | Path, event: EventRecord, *, name: str = "events.jsonl") -> Path:
    path = events_path(run_dir, name=name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event.model_dump(mode="json"), sort_keys=True))
        f.write("\n")
    return path


def load_manifest(run_dir: str | Path, *, name: str = "manifest.json") -> RunManifest | None:
    path = manifest_path(run_dir, name=name)
    if not path.exists():
        return None
    return RunManifest.model_validate(json.loads(path.read_text()))


def load_events(run_dir: str | Path, *, name: str = "events.jsonl") -> list[EventRecord]:
    path = events_path(run_dir, name=name)
    if not path.exists():
        return []
    return [EventRecord.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
