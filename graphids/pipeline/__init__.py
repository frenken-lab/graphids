"""graphids.pipeline — Orchestration layer: CLI, stages, SLURM/Dagster.

Public API:
    from graphids.pipeline import build_cli_cmd, STAGE_FNS
"""

from graphids.pipeline.stages import STAGE_FNS
from graphids.pipeline.subprocess_utils import build_cli_cmd
