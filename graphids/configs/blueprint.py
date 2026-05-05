"""Blueprint schema — the JSON array a Python plan emits.

A blueprint is an ordered list of rows. Each row is one of:
    fit / test  → carries `rendered_config` + `upstreams` + `identity`
    extract     → one-shot fusion-feature extraction (idempotent on output_dir)
    analyze     → per-checkpoint artifact generation (CKA / embeddings / …)
    cmd         → carries `command` only

Validation is one call: `BlueprintArray.model_validate(rendered_array)`.
A row that doesn't validate is a render bug; a render that fails validation
is a schema gap. Both surface here, before SLURM ever sees them.

The training-config sub-shape (`RenderedConfig`/`ClassPath`/`TrainerCfg`)
is also typed and frozen — typo'd fields raise :class:`pydantic.ValidationError`
at compose time rather than failing inside `_instantiate` with a confusing
``TypeError``. ``init_args`` stay as ``dict[str, Any]`` (free-form class kwargs).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassPath(_StrictModel):
    """A ``{class_path, init_args}`` instantiation block.

    ``init_args`` stays free-form (``dict[str, Any]``) so per-class kwargs
    don't need to be enumerated here. Nested ``class_path`` blocks inside
    ``init_args`` (e.g. ``loss_fn``, ``schedule``, ``difficulty``) are
    re-validated via Pydantic's recursive validation when typed, or stay
    as dicts when they're read by ``_instantiate``'s recursive descent.
    """

    class_path: str
    init_args: dict[str, Any] = Field(default_factory=dict)


class TrainerCfg(_StrictModel):
    """``pl.Trainer`` kwargs as rendered by the composer.

    ``callbacks`` here mirrors the top-level ``callbacks`` dict for
    jsonnet-era parity; ``orchestrate._build`` reads the dict, not this
    list. ``log_every_n_steps`` is fusion-only (``None`` elsewhere).
    """

    accelerator: str
    devices: str | int
    precision: str
    max_epochs: int
    gradient_clip_val: float | None
    callbacks: list[ClassPath]
    default_root_dir: str
    log_every_n_steps: int | None = None


class RenderedConfig(_StrictModel):
    """Composer output. Typo'd field access raises ValidationError."""

    model: ClassPath
    data: ClassPath
    callbacks: dict[str, ClassPath]
    trainer: TrainerCfg
    seed_everything: int


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
    rendered_config: RenderedConfig
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
