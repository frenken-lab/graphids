"""Per-checkpoint artifact generation.

Distinct from :mod:`graphids.analysis`, which owns cross-run statistical
comparison from the MLflow catalog (no torch). This package owns the
single-checkpoint, torch-loaded artifact pipeline (embeddings, attention,
CKA, loss landscape, fusion policy) — driven by ``AnalyzeRow``.
"""

from ._dispatch import ARTIFACTS, default_toggles_for, expected_outputs  # noqa: F401
from .analyzer import MANIFEST_NAME, Analyzer  # noqa: F401
