"""Analyzer invocation — single-checkpoint analysis + manifest sidecar.

Moved here from ``orchestrate/analyze.py``: the verb "run one analysis
against one checkpoint" is a core/analysis concern, not an
orchestration concern. ``ANALYZABLE_MODEL_TYPES`` lives alongside so
orchestration callers don't have to maintain a parallel constant that
drifts from the analyzer's actual support matrix.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphids._otel import get_logger
from graphids.core.analysis.schemas import AnalysisSpec

log = get_logger(__name__)

ANALYSIS_MANIFEST_NAME = "analysis_manifest.json"
ANALYZABLE_MODEL_TYPES: frozenset[str] = frozenset({"vgae", "dgi", "gat"})


def analysis_spec_for(
    ckpt_file: Path,
    *,
    dataset: str,
    model_type: str,
    seed: int,
) -> AnalysisSpec:
    """Build the canonical ``AnalysisSpec`` for a stage checkpoint.

    Owns the ``{run_dir}/artifacts`` layout convention so orchestration
    callers don't have to reconstruct ``ckpt_file.parent.parent`` path
    arithmetic inline.
    """
    return AnalysisSpec(
        ckpt_path=str(ckpt_file),
        dataset=dataset,
        model_type=model_type,
        output_dir=str(ckpt_file.resolve().parent.parent / "artifacts"),
        seed=seed,
    )


def run_single_analysis(spec: AnalysisSpec) -> None:
    """Run the analyzer for one checkpoint and write a manifest sidecar."""
    from graphids.core.analysis.analyzer import Analyzer
    from graphids.core.analysis.schemas import expected_outputs

    Analyzer(**spec.model_dump(exclude={"metadata"})).run()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_outputs(spec)
    existing = [name for name in expected if (output_dir / name).exists()]
    manifest = {
        "contract": spec.CONTRACT_NAME,
        "version": spec.CONTRACT_VERSION,
        "dataset": spec.dataset,
        "checkpoint_path": spec.ckpt_path,
        "output_dir": str(output_dir),
        "expected_outputs": list(expected),
        "existing_outputs": existing,
    }
    (output_dir / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
