"""Shared analysis spec/output helpers for orchestrated analysis assets."""

from __future__ import annotations

from pathlib import Path

from graphids.core.analysis.schemas import AnalysisSpec, expected_outputs
from graphids.orchestrate.planning import StageConfig

ANALYSIS_MANIFEST_NAME = "analysis_manifest.json"
ANALYSIS_SUPPORTED_MODELS = frozenset({"vgae", "gat", "dgi"})


def supports_analysis(model_type: str) -> bool:
    """Return True if the model family has supported analyzer outputs."""
    return model_type in ANALYSIS_SUPPORTED_MODELS


def analysis_flags_for(model_type: str) -> dict[str, bool]:
    """Default analyzer artifact toggles per model family."""
    return {
        "embeddings": supports_analysis(model_type),
        "attention": False,
        "cka": False,
        "landscape": False,
        "fusion_policy": False,
    }


def build_analysis_spec(
    *, cfg: StageConfig, dataset: str, seed: int, ckpt_path: str
) -> AnalysisSpec:
    """Construct one analysis spec from checkpoint and asset metadata."""
    output_dir = Path(ckpt_path).resolve().parent.parent / "artifacts"
    return AnalysisSpec(
        ckpt_path=ckpt_path,
        dataset=dataset,
        model_type=cfg.model_type,
        output_dir=str(output_dir),
        seed=seed,
        **analysis_flags_for(cfg.model_type),
    )


def output_status(spec: AnalysisSpec) -> tuple[tuple[str, ...], list[str]]:
    """Return expected output names and currently existing output names."""
    output_dir = Path(spec.output_dir)
    expected = expected_outputs(spec)
    existing = [name for name in expected if (output_dir / name).exists()]
    return expected, existing
