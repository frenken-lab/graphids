"""Plan schema — the JSON object a Python plan emits.

A plan wraps an ordered list of rows. Each row is one of:
    fit / test  → carries `rendered_config` + `upstreams` + `identity`
    extract     → one-shot fusion-feature extraction (idempotent on output_dir)
    analyze     → per-checkpoint artifact generation (CKA / embeddings / …)
    cache       → one-shot dataset cache build (vocab scan + windowing)

Validation is one call: `Plan.model_validate(rendered_object)`.
A row that doesn't validate is a render bug; a render that fails validation
is a schema gap. Both surface here, before SLURM ever sees them.

The training-config sub-shape (`RenderedConfig`/`ClassPath`/`TrainerCfg`)
is also typed and frozen — typo'd fields raise :class:`pydantic.ValidationError`
at compose time rather than failing inside `_instantiate` with a confusing
``TypeError``. ``init_args`` stay as ``dict[str, Any]`` (free-form class kwargs).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    reload_dataloaders_every_n_epochs: int = 0


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
    """Fit or test row — carries a fully-rendered training config.

    Reproduction-contract fields (``plan_module`` + ``git_sha``) are
    threaded by ``graphids run`` and surface as MLflow tags. Together
    with ``plan_id`` and ``meta.dataset`` / ``meta.seed`` they let
    ``git checkout <git_sha> && graphids run <plan_module> --dataset X
    --seed Y --filter <name>`` regenerate this exact row deterministically.
    """

    name: str
    action: Literal["fit", "test"]
    plan_id: str
    plan_module: str
    git_sha: str
    identity: Identity
    meta: Meta
    rendered_config: RenderedConfig
    upstreams: list[Upstream] = Field(default_factory=list)
    resources: Resources


class CacheRow(_StrictModel):
    """One-shot dataset cache build — vocab scan + windowing into PyG tensors.

    Idempotent: ``BaseGraphSource.build()`` short-circuits if the cache
    partition for ``(dataset, vocab_scope)`` is already present. CPU-only;
    no model, no MLflow run.
    """

    name: str
    action: Literal["cache"]
    plan_id: str
    dataset: str
    vocab_scope: Literal["train", "all"] = "train"
    seed: int = 42
    window_size: int = 100
    stride: int = 100
    val_fraction: float = 0.2
    resources: Resources


class ExtractRow(_StrictModel):
    """One-shot extraction of fusion features from upstream model ckpts.

    Idempotent: ``run_row`` short-circuits when a valid cache already exists
    at ``output_dir``. Cached outputs are reused across N fusion fit/test
    rows via Parsl ``--depends-on-afterok`` chaining.
    """

    name: str
    action: Literal["extract"]
    plan_id: str
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
    plan_id: str
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


Row = TrainRow | CacheRow | ExtractRow | AnalyzeRow


class Plan(_StrictModel):
    """Top-level rendered plan — the JSON object ``graphids run`` writes.

    Wraps the row array with grouping metadata. Every row carries the
    same ``plan_id`` so a single render can be queried as a unit
    (MLflow tag, sbatch ``--comment``, sacct lookup). ``plan_args``
    captures the inputs to ``build()`` so a TUI / re-render can
    reproduce or extend the plan without re-typing arguments.
    """

    plan_id: str
    plan_module: str
    plan_args: dict[str, Any]
    created_at: str
    rows: list[Row]

    def __iter__(self):  # type: ignore[override]
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Row:
        return self.rows[i]
