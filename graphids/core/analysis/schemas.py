"""Analysis layer schemas — ``AnalysisSpec`` + ``AnalysisContract``.

Lives next to ``analyzer.py`` because ``AnalysisSpec`` is the typed view
of ``Analyzer.__init__`` kwargs. The orchestrator produces an
``AnalysisSpec`` via ``AnalysisContract`` and hands it off to the
analysis runner.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphids.orchestrate.contracts import ContractEnvelope


class AnalysisSpec(BaseModel):
    """Canonical execution input for analyzer artifact generation."""

    model_config = ConfigDict(extra="forbid")

    ckpt_path: str
    dataset: str
    model_type: str
    output_dir: str

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

    vgae_ckpt_path: str = ""
    gat_ckpt_path: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisContract:
    """Single class owning AnalysisSpec contract operations."""

    CONTRACT_NAME = "graphids.analysis_spec"
    CONTRACT_VERSION = 1

    @classmethod
    def to_dict(cls, spec: AnalysisSpec) -> dict[str, Any]:
        return spec.model_dump(mode="json")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AnalysisSpec:
        return AnalysisSpec(**payload)

    @classmethod
    def to_envelope(
        cls,
        spec: AnalysisSpec,
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
    def from_envelope(cls, envelope_dict: dict[str, Any]) -> AnalysisSpec:
        envelope = ContractEnvelope(**envelope_dict)
        cls._validate_envelope(envelope)
        return cls.from_dict(envelope.payload)

    @classmethod
    def expected_outputs(cls, spec: AnalysisSpec) -> tuple[str, ...]:
        outputs: list[str] = []
        if spec.embeddings:
            outputs.append("embeddings.npz")
        if spec.attention:
            outputs.append("attention_weights.npz")
        if spec.cka:
            outputs.append("cka.json")
        if spec.landscape:
            outputs.append(f"loss_landscape_{spec.model_type}.parquet")
        if spec.fusion_policy:
            outputs.append("dqn_policy.json")
        return tuple(outputs)
