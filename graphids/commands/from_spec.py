"""Spec-based command entrypoint (dagster → SLURM transport).

Single entry point that deserializes a canonical spec file and dispatches
to train, test, or analyze based on ``--phase``:

    python -m graphids from-spec --phase train   --spec-file /tmp/spec.json
    python -m graphids from-spec --phase test    --spec-file /tmp/spec.json
    python -m graphids from-spec --phase analyze --spec-file /tmp/spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphids.core.contracts import AnalysisContract, TrainingContract
from graphids.core.train_entrypoint import run_test_from_spec, run_training_from_spec


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run training/test/analyze from a canonical spec (dagster transport)",
    )
    parser.add_argument("--phase", required=True, choices=("train", "test", "analyze"))
    parser.add_argument("--spec-file", required=True)
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.spec_file).read_text())

    if args.phase == "train":
        run_training_from_spec(TrainingContract.from_envelope(payload))
        return

    if args.phase == "test":
        run_test_from_spec(TrainingContract.from_envelope(payload))
        return

    # phase == "analyze"
    from graphids.core.artifacts import Analyzer
    from graphids.orchestrate.analysis import ANALYSIS_MANIFEST_NAME, output_status

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
