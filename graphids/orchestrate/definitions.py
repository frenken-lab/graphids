"""Dagster definitions entry point.

Instantiates SlurmTrainingComponent and builds Definitions.
Discovered by dg CLI via pyproject.toml code_location_target_module.
"""

import os

from dagster.components import build_defs_for_component

from graphids.orchestrate.component import SlurmTrainingComponent

component = SlurmTrainingComponent(
    dry_run=os.environ.get("KD_GAT_DRY_RUN", "").lower() in ("1", "true"),
)

defs = build_defs_for_component(component)
