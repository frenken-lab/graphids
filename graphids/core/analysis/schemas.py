"""Analysis layer schemas — ``AnalysisSpec`` + per-model artifact dispatch.

Lives next to ``analyzer.py`` because ``AnalysisSpec`` is the typed view
of ``Analyzer.__init__`` kwargs. The orchestrator produces an
``AnalysisSpec`` and hands it off to the analysis runner.

``ARTIFACTS_BY_MODEL_TYPE`` is the single dispatch table for "which
artifacts fire for which model type" — consumed by both the CLI
(``cli.analysis.analyze``) and the pipeline driver
(``runner.analysis_spec_for``). A new model family adds one row here,
not a new jsonnet stage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Per-model artifact toggles. Fusion's ``fusion_policy`` flag is only
# flipped on when upstream VGAE+GAT checkpoints are available (enforced
# by ``analysis_spec_for`` + the AnalysisSpec conditional validator).
ARTIFACTS_BY_MODEL_TYPE: dict[str, dict[str, bool | int | float]] = {
    "vgae": {
        "embeddings": True,
        "landscape": True,
        "landscape_resolution": 51,
        "landscape_scale": 1.0,
    },
    "dgi": {
        "embeddings": True,
        "landscape": True,
        "landscape_resolution": 51,
        "landscape_scale": 1.0,
    },
    "gat": {
        "embeddings": True,
        "attention": True,
        "cka": True,
        "landscape": True,
    },
    "fusion": {
        "embeddings": False,
    },
}

# Map class_path substrings → model_type for checkpoint self-description.
# Ordered most-specific first so ``vgae_module`` beats a hypothetical
# catch-all. The ``class_path`` itself is written by ``_build_checkpoint``
# and read by ``safe_load_checkpoint`` — no drift risk.
_CLASS_PATH_TO_MODEL_TYPE: tuple[tuple[str, str], ...] = (
    ("vgae_module", "vgae"),
    ("autoencoder.vgae.VGAE", "vgae"),
    ("dgi_module", "dgi"),
    ("autoencoder.dgi.DGI", "dgi"),
    ("gat_module", "gat"),
    ("supervised.gat.GAT", "gat"),
    ("BanditFusion", "fusion"),
    ("DQNFusion", "fusion"),
    ("MLPFusion", "fusion"),
    ("WeightedAvg", "fusion"),
)


def derive_model_type(ckpt_path: str | Path) -> str:
    """Read the checkpoint's ``class_path`` and map it to an analyzer model_type."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    from graphids._fs import atomic_load

    ckpt = atomic_load(path, map_location="cpu", weights_only=True)
    class_path = ckpt.get("class_path")
    if not class_path:
        raise KeyError(
            f"Checkpoint {path} missing 'class_path'. Re-train with the current "
            "callbacks.ModelCheckpoint to produce self-describing checkpoints."
        )
    for needle, mt in _CLASS_PATH_TO_MODEL_TYPE:
        if needle in class_path:
            return mt
    raise ValueError(f"Unrecognized class_path {class_path!r} for analysis dispatch")


class AnalysisSpec(BaseModel):
    """Canonical execution input for analyzer artifact generation."""

    model_config = ConfigDict(extra="forbid")

    CONTRACT_NAME: ClassVar[str] = "graphids.analysis_spec"
    CONTRACT_VERSION: ClassVar[int] = 1

    ckpt_path: str
    dataset: str
    model_type: Literal["vgae", "dgi", "gat", "fusion"]
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
        if self.cka and self.model_type != "gat":
            # CKA uses model(g, return_intermediate=True), only GATWithJK supports this
            raise ValueError(
                f"cka=true only supported for model_type='gat', got {self.model_type!r}"
            )
        if self.fusion_policy and not self.vgae_ckpt_path:
            raise ValueError("fusion_policy=true requires vgae_ckpt_path")
        if self.fusion_policy and not self.gat_ckpt_path:
            raise ValueError("fusion_policy=true requires gat_ckpt_path")
        return self


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
