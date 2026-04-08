"""Analysis layer schemas — ``AnalysisSpec`` + envelope helpers.

Lives next to ``analyzer.py`` because ``AnalysisSpec`` is the typed view
of ``Analyzer.__init__`` kwargs. The orchestrator produces an
``AnalysisSpec`` and hands it off to the analysis runner.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graphids.contracts import from_envelope as _from_envelope
from graphids.contracts import to_envelope as _to_envelope


class AnalysisSpec(BaseModel):
    """Canonical execution input for analyzer artifact generation."""

    model_config = ConfigDict(extra="forbid")

    CONTRACT_NAME: ClassVar[str] = "graphids.analysis_spec"
    CONTRACT_VERSION: ClassVar[int] = 1

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

    @model_validator(mode="after")
    def _validate_conditional_deps(self) -> AnalysisSpec:
        if self.cka and not self.cka_teacher_ckpt:
            raise ValueError("cka=true requires cka_teacher_ckpt")
        if self.fusion_policy and not self.vgae_ckpt_path:
            raise ValueError("fusion_policy=true requires vgae_ckpt_path")
        if self.fusion_policy and not self.gat_ckpt_path:
            raise ValueError("fusion_policy=true requires gat_ckpt_path")
        return self


def to_envelope(spec: AnalysisSpec, *, metadata: dict[str, Any] | None = None):
    return _to_envelope(spec, metadata=metadata)


def from_envelope(envelope: dict[str, Any]) -> AnalysisSpec:
    return _from_envelope(envelope, AnalysisSpec)


def expected_outputs(spec: AnalysisSpec) -> tuple[str, ...]:
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
