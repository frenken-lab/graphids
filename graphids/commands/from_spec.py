"""Spec-based command entrypoints (dagster → SLURM transport).

Three commands that deserialize a canonical spec file and dispatch to the
corresponding training/test/analyze code path:

    python -m graphids train-from-spec   --spec-file /tmp/spec.json
    python -m graphids test-from-spec    --spec-file /tmp/spec.json
    python -m graphids analyze-from-spec --spec-file /tmp/spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphids.core.contracts import AnalysisContract, TrainingContract
from graphids.core.train_entrypoint import run_test_from_spec, run_training_from_spec


def _load_payload(argv: list[str], description: str) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spec-file", required=True)
    args = parser.parse_args(argv)
    return json.loads(Path(args.spec_file).read_text())


def main_train(argv: list[str]) -> None:
    payload = _load_payload(argv, "Run fit from canonical TrainingSpec")
    run_training_from_spec(TrainingContract.from_envelope(payload))


def main_test(argv: list[str]) -> None:
    payload = _load_payload(argv, "Run test from canonical TrainingSpec")
    run_test_from_spec(TrainingContract.from_envelope(payload))


def main_analyze(argv: list[str]) -> None:
    from graphids.core.artifacts import Analyzer
    from graphids.orchestrate.analysis import ANALYSIS_MANIFEST_NAME, output_status

    payload = _load_payload(argv, "Run analyzer from canonical AnalysisSpec")
    spec = AnalysisContract.from_envelope(payload)

    runner_payload = spec.model_dump(mode="python")
    runner_payload.pop("metadata", None)
    Analyzer(**runner_payload).run()

    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected, existing = output_status(spec)
    manifest = {
        "contract": AnalysisContract.CONTRACT_NAME,
        "version": AnalysisContract.CONTRACT_VERSION,
        "asset": spec.metadata.get("asset_name", "unknown"),
        "dataset": spec.dataset,
        "seed": spec.seed,
        "checkpoint_path": spec.ckpt_path,
        "output_dir": str(output_dir),
        "expected_outputs": list(expected),
        "existing_outputs": existing,
    }
    (output_dir / ANALYSIS_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
