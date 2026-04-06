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

from graphids.contracts import from_envelope as _from_envelope
from graphids.contracts import to_envelope as _to_envelope

from graphids.config.constants import PROJECT_ROOT

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

# Stage -> jsonnet filename (all fusion methods share one stage file;
# method dispatch happens inside fusion.jsonnet via the fusion_method TLA).
_STAGE_JSONNET: dict[str, str] = {
    "autoencoder": "autoencoder.jsonnet",
    "supervised": "supervised.jsonnet",
    "fusion": "fusion.jsonnet",
}


def to_envelope(spec: TrainingSpec, *, metadata: dict[str, Any] | None = None):
    return _to_envelope(spec, metadata=metadata)


def from_envelope(payload: dict[str, Any]) -> TrainingSpec:
    return _from_envelope(payload, TrainingSpec)


def normalize_scale(scale: str) -> str:
    if scale not in {"small", "large"}:
        raise ValueError(f"Unsupported scale '{scale}'. Expected: small or large.")
    return scale


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

    # model_init_overrides carry planner-derived per-model knobs
    # (conv_type, loss_fn, variational). Map the ones the stage
    # accepts; stringified bool "true"/"false" re-cast to real bool.
    for key in ("conv_type", "variational", "loss_fn"):
        if key in stage_cfg.model_init_overrides:
            val = stage_cfg.model_init_overrides[key]
            if key == "variational":
                tla[key] = val in (True, "true", "True")
            else:
                tla[key] = val

    if stage_cfg.stage == "fusion":
        tla["fusion_method"] = stage_cfg.resource_model or stage_cfg.model_type

    # Upstream checkpoint paths wire to the correct TLA based on which
    # upstream model family produced them. Fusion reads gat_ckpt_path;
    # GAT students with VGAE/DGI teachers read vgae_ckpt_path.
    for upstream_asset, ckpt in upstream_ckpts.items():
        family = upstream_model_families.get(upstream_asset)
        if family == "unsupervised":
            tla["vgae_ckpt_path"] = ckpt
        elif family == "supervised":
            tla["gat_ckpt_path"] = ckpt

    # KD distillation config (null = no KD). Jsonnet stages accept
    # `distillation_config` as the TLA name.
    if kd_overrides:
        tla["distillation_config"] = dict(kd_overrides)
    else:
        tla["distillation_config"] = None

    if ckpt_path is not None:
        tla["ckpt_path"] = ckpt_path

    return tla


__all__ = [
    "TrainingSpec",
    "build_tla_dict",
    "from_envelope",
    "normalize_scale",
    "resolve_jsonnet_path",
    "to_envelope",
]
