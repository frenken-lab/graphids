"""Shared analysis runner for orchestrated analysis assets."""

from __future__ import annotations

import json
from pathlib import Path

from graphids.core.analysis.schemas import AnalysisSpec, expected_outputs

ANALYSIS_MANIFEST_NAME = "analysis_manifest.json"


def run_analysis(spec: AnalysisSpec) -> None:
    """Run the analyzer and write a manifest sidecar.

    Called by the Monarch actor ``eval_stage`` endpoint.
    """
    from graphids.core.analysis.analyzer import Analyzer

    Analyzer(**spec.model_dump(exclude={"metadata"})).run()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_outputs(spec)
    existing = [name for name in expected if (output_dir / name).exists()]
    manifest = {
        "contract": AnalysisSpec.CONTRACT_NAME,
        "version": AnalysisSpec.CONTRACT_VERSION,
        "dataset": spec.dataset,
        "checkpoint_path": spec.ckpt_path,
        "output_dir": str(output_dir),
        "expected_outputs": list(expected),
        "existing_outputs": existing,
    }
    (output_dir / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
