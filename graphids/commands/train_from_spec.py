"""Run training from a serialized TrainingSpec payload.

Usage:
    python -m graphids train-from-spec --spec-file /path/to/spec.json
"""

from __future__ import annotations

import argparse

from graphids.commands._spec_payload import load_payload
from graphids.core.train_entrypoint import run_training_from_payload


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Run fit from canonical TrainingSpec")
    parser.add_argument("--spec-file", required=True)
    args = parser.parse_args(argv)

    payload = load_payload(args.spec_file)
    run_training_from_payload(payload)
