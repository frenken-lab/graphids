"""Programmatic analyzer entrypoint from canonical AnalysisSpec."""

from __future__ import annotations

from typing import Any

from graphids.core.artifacts import Analyzer
from graphids.core.contracts import AnalysisContract, AnalysisSpec


def run_analysis_from_spec(spec: AnalysisSpec) -> None:
    """Execute one analyzer run from canonical spec."""
    payload = spec.model_dump(mode="python")
    payload.pop("metadata", None)
    Analyzer(**payload).run()


def run_analysis_from_payload(payload: dict[str, Any]) -> None:
    run_analysis_from_spec(AnalysisContract.from_envelope(payload))
