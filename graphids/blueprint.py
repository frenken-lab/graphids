"""Blueprint schema — the JSON array emitted by `plan.jsonnet`.

A blueprint is an ordered list of rows. Each row is one of:
    fit / test  → carries `rendered_config` + `upstreams` + `identity`
    cmd         → carries `command` only

Validation is one call: `BlueprintArray.model_validate(rendered_array)`.
A row that doesn't validate is a render bug; a render that fails validation
is a schema gap. Both surface here, before SLURM ever sees them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Identity(_StrictModel):
    run_name: str
    run_dir: str
    jobname: str


class Meta(_StrictModel):
    """Structured per-row identity. Mirrors a preset's `_meta` block."""

    group: str
    variant: str
    dataset: str
    seed: int
    model_type: str
    scale: str


class Upstream(_StrictModel):
    role: str
    ckpt_path: str
    ckpt_tla: str


class Resources(_StrictModel):
    mode: Literal["gpu", "cpu"]
    length: Literal["short", "long"]


class TrainRow(_StrictModel):
    """Fit or test row — carries a fully-rendered training config."""

    name: str
    action: Literal["fit", "test"]
    identity: Identity
    meta: Meta
    rendered_config: dict[str, Any]
    upstreams: list[Upstream] = Field(default_factory=list)
    resources: Resources


class CmdRow(_StrictModel):
    """Non-training row — runs an arbitrary shell command on a SLURM node."""

    name: str
    action: Literal["cmd"]
    command: str
    resources: Resources


Row = TrainRow | CmdRow


class BlueprintArray(RootModel[list[Row]]):
    """Top-level blueprint — array of rows in execution order."""

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, i: int) -> Row:
        return self.root[i]
