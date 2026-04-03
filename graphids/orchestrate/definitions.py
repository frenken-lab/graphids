"""Dagster definitions entry point.

Instantiates SlurmTrainingComponent and builds Definitions.
Discovered by dg CLI via pyproject.toml code_location_target_module.

Structlog is configured here (not __main__.py) because dagster workers
import this module directly. Under SLURM: JSONL to a per-run log file.
Otherwise: ConsoleRenderer for validation / dg list defs.
"""

import os

import structlog

from graphids.config import SLURM_LOG_DIR

# ---------------------------------------------------------------------------
# Structlog configuration — process-global, affects resolve.py + slurm.py too
# ---------------------------------------------------------------------------
_PROCESSORS = [
    structlog.contextvars.merge_contextvars,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
]

_slurm_job = os.environ.get("SLURM_JOB_ID")
if _slurm_job:
    _log_path = f"{SLURM_LOG_DIR}/orchestrator_{_slurm_job}.jsonl"
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)
    structlog.configure(
        processors=[*_PROCESSORS, structlog.processors.JSONRenderer()],
        logger_factory=structlog.WriteLoggerFactory(
            file=open(_log_path, "a"),  # noqa: SIM115
        ),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(slurm_job_id=_slurm_job)
else:
    structlog.configure(
        processors=[*_PROCESSORS, structlog.dev.ConsoleRenderer()],
        cache_logger_on_first_use=True,
    )

# ---------------------------------------------------------------------------

from dagster.components import build_defs_for_component  # noqa: E402

from graphids.orchestrate.component import SlurmTrainingComponent  # noqa: E402

component = SlurmTrainingComponent(
    dry_run=os.environ.get("KD_GAT_DRY_RUN", "").lower() in ("1", "true"),
)

defs = build_defs_for_component(component)
