"""Per-asset Dagster Config for launch-time overrides.

Phase 5: separates launch-time behavioral knobs from planner-derived
identity fields (which stay in StageConfig). These fields are overridable
in the Dagster UI/CLI at materialization time.
"""

from __future__ import annotations

import dagster as dg


class TrainingAssetConfig(dg.Config):
    """Launch-time overridable knobs for training assets."""

    run_test: bool = True
    run_analysis: bool = True
    dry_run: bool = False
