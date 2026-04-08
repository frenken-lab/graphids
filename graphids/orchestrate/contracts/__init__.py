"""Orchestration → execution boundary specs and helpers.

Planner-side typed boundary values for the training path:

- ``TrainingSpec`` — canonical training input handed from orchestration
  (dagster, CLI ``from-spec``) to the execution layer.
- Envelope helpers live in ``graphids.contracts``.

``AnalysisSpec`` + envelope helpers live next to ``analyzer.py`` in
``graphids.core.analysis.schemas`` because they validate analyzer init
kwargs — the schema belongs with its consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from graphids.config.constants import PROJECT_ROOT
from graphids.config.topology import TOPOLOGY
from graphids.contracts import from_envelope as _from_envelope
from graphids.contracts import to_envelope as _to_envelope

if TYPE_CHECKING:
    from graphids.orchestrate.planning.shared import StageConfig


CONFIGS_DIR = PROJECT_ROOT / "configs"


# -----------------------------------------------------------------------------
# Pydantic models — boundary values
# -----------------------------------------------------------------------------


class TrainingSpec(BaseModel):
    """Canonical execution input shared by CLI and orchestrators.

    Carries a single ``jsonnet_path`` + typed ``jsonnet_tla`` dict. Everything
    the stage function needs to render a fully-merged config is in
    ``jsonnet_tla``; ``build_tla_dict`` is the only site that constructs it.
    """

    model_config = ConfigDict(extra="forbid")

    CONTRACT_NAME: ClassVar[str] = "graphids.training_spec"
    CONTRACT_VERSION: ClassVar[int] = 2  # bumped for jsonnet_path/jsonnet_tla fields

    stage: str
    model_family: str
    scale: str
    dataset: str
    seed: int
    run_dir: str
    jsonnet_path: str
    jsonnet_tla: dict[str, Any] = Field(default_factory=dict)
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    upstream_ckpt_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Populated from Dagster asset I/O at materialization time, not from config.",
    )
    upstream_model_families: dict[str, str] = Field(default_factory=dict)


_STAGES_DIR = CONFIGS_DIR / "stages"

# Convention: stage name == jsonnet filename (topology.py validates existence).
_STAGE_JSONNET: dict[str, str] = {s: f"{s}.jsonnet" for s in TOPOLOGY.stages}


def to_envelope(spec: TrainingSpec, *, metadata: dict[str, Any] | None = None):
    return _to_envelope(spec, metadata=metadata)


def from_envelope(payload: dict[str, Any]) -> TrainingSpec:
    return _from_envelope(payload, TrainingSpec)


def resolve_jsonnet_path(stage: str) -> str:
    """Return the absolute path to the jsonnet file for a stage."""
    filename = _STAGE_JSONNET.get(stage)
    if filename is None:
        raise ValueError(
            f"No jsonnet stage file for stage={stage!r}. Known: {sorted(_STAGE_JSONNET)}"
        )
    return str(_STAGES_DIR / filename)


def build_tla_dict(
    stage_cfg: StageConfig,
    *,
    dataset: str,
    seed: int,
    run_dir: str,
    upstream_ckpts: dict[str, str],
    upstream_model_families: dict[str, str],
    kd_overrides: dict[str, Any] | None = None,
    trainer_overrides: dict[str, str] | None = None,
    stage_overrides: dict[str, str] | None = None,
    ckpt_path: str | None = None,
) -> dict[str, Any]:
    """Build the typed TLA dict consumed by the stage's jsonnet function.

    Values are real typed primitives (ints stay ints, bools stay bools,
    lists stay lists) — jsonnet's ``--tla-code`` JSON-encodes each value
    so round-trip is exact.
    """
    tla: dict[str, Any] = {
        "dataset": dataset,
        "seed": seed,
        "run_dir": run_dir,
        "scale": stage_cfg.scale,
        "trainer_overrides": dict(trainer_overrides or {}),
        "stage_overrides": dict(stage_overrides or {}),
    }

    # stage_tlas in topology.json declares every non-common TLA the
    # stage's jsonnet function accepts. Gate ALL optional TLAs through
    # it — no if/elif per stage, no unknown-parameter crashes.
    stage_def = TOPOLOGY.stages.get(stage_cfg.stage)
    accepted = set(stage_def.stage_tlas) if stage_def else set()

    # Model knobs from planner (conv_type, loss_fn, variational)
    for key in ("conv_type", "variational", "loss_fn"):
        if key in stage_cfg.model_init_overrides and key in accepted:
            val = stage_cfg.model_init_overrides[key]
            if key == "variational":
                tla[key] = val in (True, "true", "True")
            else:
                tla[key] = val

    if "fusion_method" in accepted:
        tla["fusion_method"] = stage_cfg.resource_model or stage_cfg.model_type

    # Upstream checkpoint paths
    for upstream_asset, ckpt in upstream_ckpts.items():
        family = upstream_model_families.get(upstream_asset)
        if family == "unsupervised" and "vgae_ckpt_path" in accepted:
            tla["vgae_ckpt_path"] = ckpt
        elif family == "supervised" and "gat_ckpt_path" in accepted:
            tla["gat_ckpt_path"] = ckpt

    # KD distillation config
    if "distillation_config" in accepted:
        tla["distillation_config"] = dict(kd_overrides) if kd_overrides else None

    if ckpt_path is not None and "ckpt_path" in accepted:
        tla["ckpt_path"] = ckpt_path

    return tla


__all__ = [
    "TrainingSpec",
    "build_tla_dict",
    "from_envelope",
    "resolve_jsonnet_path",
    "to_envelope",
]
