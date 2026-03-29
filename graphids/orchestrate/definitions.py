"""Dagster definitions entry point.

Instantiates SlurmTrainingComponent and builds Definitions from it.
Uses build_defs_for_component for standalone component loading.
"""

import os

from dagster.components import build_defs_for_component

from graphids.components.slurm_training_component import SlurmTrainingComponent

component = SlurmTrainingComponent(
    lake_root=os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns"),
    user=os.environ.get("USER", "unknown"),
    dry_run=os.environ.get("KD_GAT_DRY_RUN", "").lower() in ("1", "true"),
)

defs = build_defs_for_component(component)
