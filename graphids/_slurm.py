"""Single indirection point for SLURM interaction.

Every module imports from here, never from pyslurm or os.environ directly.
Swap pyslurm for subprocess fallback by changing THIS file only.

    from graphids._slurm import slurm_account, slurm_cpus_per_task, job_accounting
"""

from __future__ import annotations

import os

from pyslurm.db import Job as _DbJob, JobFilter as _JobFilter, Jobs as _DbJobs
from pyslurm.utils.ctime import timestr_to_mins, timestr_to_secs
from pyslurm.utils.helpers import dehumanize


# ---------------------------------------------------------------------------
# Env vars — SLURM-injected (runtime only) and project settings
# ---------------------------------------------------------------------------

def slurm_account() -> str:
    from graphids.config.settings import get_settings
    return get_settings().slurm_account


def slurm_job_id() -> str | None:
    return os.environ.get("SLURM_JOB_ID")


def slurm_cpus_per_task() -> int | None:
    val = os.environ.get("SLURM_CPUS_PER_TASK")
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Accounting — pyslurm.db backed
# ---------------------------------------------------------------------------

def job_accounting(job_id: int) -> dict[str, str | int]:
    """Return ``{job_id, wall_time, peak_rss}`` for a completed job."""
    try:
        job = _DbJob.load(job_id)
    except Exception:
        return {"job_id": job_id, "wall_time": "", "peak_rss": ""}

    wall = job.elapsed_time or 0
    h, rem = divmod(wall, 3600)
    m, s = divmod(rem, 60)
    wall_str = f"{h}:{m:02d}:{s:02d}"

    rss = job.stats.resident_memory if job.stats else 0
    rss_gib = rss / (1024 ** 3) if rss else 0
    rss_str = f"{rss_gib:.2f}G" if rss else ""

    return {"job_id": job_id, "wall_time": wall_str, "peak_rss": rss_str}


def load_jobs(
    job_ids: list[str] | list[int] | None = None,
    *,
    user: str | None = None,
    start_time: str | None = None,
) -> dict:
    """Load jobs from slurmdbd. Returns {job_id: db.Job}."""
    filt = _JobFilter()
    if job_ids:
        filt.ids = [int(j) for j in job_ids]
    if user:
        filt.users = [user]
    if start_time:
        filt.start_time = start_time
    try:
        return dict(_DbJobs.load(filt))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Re-exports — utilities consumers need without importing pyslurm directly
# ---------------------------------------------------------------------------

__all__ = [
    "dehumanize",
    "job_accounting",
    "load_jobs",
    "slurm_account",
    "slurm_cpus_per_task",
    "slurm_job_id",
    "timestr_to_mins",
    "timestr_to_secs",
]
