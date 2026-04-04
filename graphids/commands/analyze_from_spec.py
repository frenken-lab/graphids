"""Run analyzer from a serialized AnalysisSpec payload.

Usage:
    python -m graphids analyze-from-spec --spec-file /path/to/spec.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from graphids.commands._spec_payload import load_payload
from graphids.core.analyze_entrypoint import run_analysis_from_payload
from graphids.core.contracts import AnalysisContract
from graphids.orchestrate.analysis import ANALYSIS_MANIFEST_NAME, output_status


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run analyzer from canonical AnalysisSpec")
    parser.add_argument("--spec-file", required=True)
    args = parser.parse_args(argv)

    payload = load_payload(args.spec_file)
    run_analysis_from_payload(payload)

    spec = AnalysisContract.from_envelope(payload)
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
