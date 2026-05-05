"""Blueprint schema — the JSON array emitted by `plan.jsonnet`.

A blueprint is an ordered list of rows. Each row is one of:
    fit / test  → carries `rendered_config` + `upstreams` + `identity`
    extract     → one-shot fusion-feature extraction (idempotent on output_dir)
    analyze     → per-checkpoint artifact generation (CKA / embeddings / …)
    cmd         → carries `command` only

Validation is one call: `BlueprintArray.model_validate(rendered_array)`.
A row that doesn't validate is a render bug; a render that fails validation
is a schema gap. Both surface here, before SLURM ever sees them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


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


class ExtractRow(_StrictModel):
    """One-shot extraction of fusion features from upstream model ckpts.

    Idempotent: ``run_row`` short-circuits when a valid cache already exists
    at ``output_dir``. Cached outputs are reused across N fusion fit/test
    rows via Parsl ``--depends-on-afterok`` chaining.
    """

    name: str
    action: Literal["extract"]
    dataset: str
    extractor_ckpts: dict[str, str]
    output_dir: str
    resources: Resources
    max_samples: int = 150_000
    max_val_samples: int = 30_000
    batch_size: int = 256
    seed: int = 42
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2


class AnalyzeRow(_StrictModel):
    """Per-checkpoint artifact generation row.

    Drives :class:`graphids.core.artifacts.Analyzer` directly — every field
    here is consumed by an artifact in
    :data:`graphids.core.artifacts.ARTIFACTS`. Add a new artifact ⇒ add one
    row to that table; add new state ⇒ add one field here. The Analyzer's
    constructor takes a row instance, not 30 kwargs.
    """

    name: str
    action: Literal["analyze"]
    resources: Resources

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

    window_size: int = 100
    stride: int = 100
    batch_size: int = 256
    seed: int = 42
    vocab_scope: str = "train"

    vgae_ckpt_path: str = ""
    gat_ckpt_path: str = ""

    @model_validator(mode="after")
    def _validate_conditional_deps(self) -> AnalyzeRow:
        if self.cka and not self.cka_teacher_ckpt:
            raise ValueError("cka=true requires cka_teacher_ckpt")
        if self.cka and self.model_type != "gat":
            # CKA needs return_intermediate=True — only GATWithJK exposes it.
            raise ValueError(
                f"cka=true only supported for model_type='gat', got {self.model_type!r}"
            )
        if self.fusion_policy and not self.vgae_ckpt_path:
            raise ValueError("fusion_policy=true requires vgae_ckpt_path")
        if self.fusion_policy and not self.gat_ckpt_path:
            raise ValueError("fusion_policy=true requires gat_ckpt_path")
        return self


Row = TrainRow | CmdRow | ExtractRow | AnalyzeRow


class BlueprintArray(RootModel[list[Row]]):
    """Top-level blueprint — array of rows in execution order."""

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, i: int) -> Row:
        return self.root[i]
