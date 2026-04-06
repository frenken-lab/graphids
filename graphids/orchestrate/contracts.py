"""Orchestration → execution boundary contracts.

Planner-side typed boundary values for the training path:

- ``TrainingSpec`` — canonical training input handed from orchestration
  (dagster, CLI ``from-spec``) to the execution layer.
- ``ContractEnvelope`` — versioned wrapper for serialized payloads.
- ``TrainingContract`` — class-based operations (pack/unpack, envelope
  roundtrip, TLA dict construction for jsonnet).

``AnalysisSpec`` + ``AnalysisContract`` live next to ``analyzer.py`` in
``graphids.core.analysis.schemas`` because they validate analyzer init
kwargs — the schema belongs with its consumer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from graphids.config.constants import PROJECT_ROOT

if TYPE_CHECKING:
    from graphids.orchestrate.shared import StageConfig


CONFIGS_DIR = PROJECT_ROOT / "configs"


# -----------------------------------------------------------------------------
# Pydantic models — boundary values
# -----------------------------------------------------------------------------


class TrainingSpec(BaseModel):
    """Canonical execution input shared by CLI and orchestrators.

    Carries a single ``jsonnet_path`` + typed ``jsonnet_tla`` dict. Everything
    the stage function needs to render a fully-merged config is in
    ``jsonnet_tla``; ``TrainingContract.build_tla_dict`` is the only site that
    constructs it.
    """

    model_config = ConfigDict(extra="forbid")

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


class ContractEnvelope(BaseModel):
    """Versioned wrapper for serialized contract payloads."""

    model_config = ConfigDict(extra="forbid")

    contract: str
    version: int
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)


# -----------------------------------------------------------------------------
# TrainingContract — class-based ops
# -----------------------------------------------------------------------------


class TrainingContract:
    """Single class owning TrainingSpec contract operations."""

    CONTRACT_NAME = "graphids.training_spec"
    CONTRACT_VERSION = 2  # bumped for jsonnet_path/jsonnet_tla fields

    _STAGES_DIR = CONFIGS_DIR / "stages"

    # Stage -> jsonnet filename (all fusion methods share one stage file;
    # method dispatch happens inside fusion.jsonnet via the fusion_method TLA).
    _STAGE_JSONNET: dict[str, str] = {
        "autoencoder": "autoencoder.jsonnet",
        "supervised": "supervised.jsonnet",
        "fusion": "fusion.jsonnet",
    }

    @classmethod
    def to_dict(cls, spec: TrainingSpec) -> dict[str, Any]:
        return spec.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TrainingSpec:
        return TrainingSpec(**payload)

    @classmethod
    def to_envelope(
        cls,
        spec: TrainingSpec,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ContractEnvelope:
        return ContractEnvelope(
            contract=cls.CONTRACT_NAME,
            version=cls.CONTRACT_VERSION,
            payload=cls.to_dict(spec),
            metadata=metadata or {},
        )

    @classmethod
    def _validate_envelope(cls, envelope: ContractEnvelope) -> None:
        if envelope.contract != cls.CONTRACT_NAME:
            raise ValueError(
                f"Unexpected contract {envelope.contract!r}; expected {cls.CONTRACT_NAME!r}"
            )
        if envelope.version != cls.CONTRACT_VERSION:
            raise ValueError(
                f"Unsupported contract version {envelope.version}; expected {cls.CONTRACT_VERSION}"
            )

    @classmethod
    def from_envelope(cls, envelope_dict: dict[str, Any]) -> TrainingSpec:
        envelope = ContractEnvelope(**envelope_dict)
        cls._validate_envelope(envelope)
        return cls.from_dict(envelope.payload)

    @classmethod
    def normalize_scale(cls, scale: str) -> str:
        if scale not in {"small", "large"}:
            raise ValueError(f"Unsupported scale '{scale}'. Expected: small or large.")
        return scale

    @classmethod
    def resolve_jsonnet_path(cls, stage: str) -> str:
        """Return the absolute path to the jsonnet file for a stage."""
        filename = cls._STAGE_JSONNET.get(stage)
        if filename is None:
            raise ValueError(
                f"No jsonnet stage file for stage={stage!r}. Known: {sorted(cls._STAGE_JSONNET)}"
            )
        return str(cls._STAGES_DIR / filename)

    @classmethod
    def build_tla_dict(
        cls,
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
