"""Run analyzer from a serialized AnalysisSpec payload.

Usage:
    python -m graphids analyze-from-spec --spec-file /path/to/spec.json
"""

from __future__ import annotations

import argparse

from graphids.commands._spec_payload import load_payload
from graphids.core.analyze_entrypoint import run_analysis_from_payload


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run analyzer from canonical AnalysisSpec")
    parser.add_argument("--spec-file", required=True)
    args = parser.parse_args(argv)

    payload = load_payload(args.spec_file)
    run_analysis_from_payload(payload)
