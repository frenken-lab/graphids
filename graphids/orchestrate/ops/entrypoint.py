"""Spec-based command operation layer (dagster → SLURM transport).

Deserializes a canonical spec envelope and dispatches to train / test /
analyze. Argparse surface lives in ``graphids.commands.from_spec``.
"""

from __future__ import annotations

import json
from pathlib import Path

from graphids.contracts import from_envelope
from graphids.core.analysis.schemas import AnalysisSpec
from graphids.core.train_entrypoint import run_test_from_spec, run_training_from_spec
from graphids.orchestrate.contracts import TrainingSpec


def run_from_spec(phase: str, spec_file: Path) -> None:
    """Load spec envelope and dispatch based on ``phase``.

    ``phase`` must be one of ``"train"`` / ``"test"`` / ``"analyze"``.
    Training and test paths delegate to ``graphids.core.train_entrypoint``.
    Analyze instantiates the Analyzer directly and writes a manifest
    sidecar next to its outputs.
    """
    payload = json.loads(spec_file.read_text())

    if phase == "train":
        run_training_from_spec(from_envelope(payload, TrainingSpec))
        return

    if phase == "test":
        run_test_from_spec(from_envelope(payload, TrainingSpec))
        return

    # phase == "analyze"
    from graphids.core.analysis.analyzer import Analyzer
    from graphids.orchestrate.analysis import ANALYSIS_MANIFEST_NAME, output_status

    spec = from_envelope(payload, AnalysisSpec)

    runner_payload = spec.model_dump(mode="python")
    runner_payload.pop("metadata", None)
    Analyzer(**runner_payload).run()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected, existing = output_status(spec)
    manifest = {
        "contract": AnalysisSpec.CONTRACT_NAME,
        "version": AnalysisSpec.CONTRACT_VERSION,
        "asset": spec.metadata.get("asset_name", "unknown"),
        "dataset": spec.dataset,
        "seed": spec.seed,
        "checkpoint_path": spec.ckpt_path,
        "output_dir": str(output_dir),
        "expected_outputs": list(expected),
        "existing_outputs": existing,
    }
    (output_dir / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
