"""Dagster definitions entry point.

Instantiates SlurmTrainingComponent and builds Definitions.
Discovered by dg CLI via pyproject.toml code_location_target_module.

Logging is configured here (not __main__.py) because dagster workers
import this module directly. Under SLURM: JSONL to a per-run log file.
Otherwise: human-readable stderr for validation / dg list defs.
"""

from graphids.config.settings import get_settings
from graphids.log import configure_logging
from graphids.slurm.env import SLURM_LOG_DIR, slurm_job_id

# ---------------------------------------------------------------------------
# Logging configuration — process-global, affects resolve.py + slurm.py too
# ---------------------------------------------------------------------------
_slurm_job = slurm_job_id()
if _slurm_job:
    configure_logging(jsonl_path=f"{SLURM_LOG_DIR}/orchestrator_{_slurm_job}.jsonl")
else:
    configure_logging()

# ---------------------------------------------------------------------------

from dagster.components import build_defs_for_component  # noqa: E402

from graphids.orchestrate.dagster.component import SlurmTrainingComponent  # noqa: E402

component = SlurmTrainingComponent(
    dry_run=get_settings().dry_run,
)

defs = build_defs_for_component(component)
