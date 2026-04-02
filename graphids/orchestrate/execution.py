"""Execution helpers for orchestrated training assets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from graphids.config import COMPLETE_MARKER
from graphids.slurm import sacct_query


def touch_complete(rd_path: Path) -> None:
    """Write the .complete marker after a successful training run.

    Uses fsync to ensure visibility on NFS before returning.
    """
    marker = rd_path / COMPLETE_MARKER
    marker.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(marker), os.O_CREAT | os.O_WRONLY, 0o664)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # fsync the directory so the new entry is durable on NFS
    dir_fd = os.open(str(marker.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def slurm_accounting_metadata(job_id: int) -> dict[str, Any]:
    """Extract wall time and peak RSS from sacct output."""
    out = sacct_query([job_id], "JobID,Elapsed,MaxRSS", units="G")
    wall, rss = "", ""
    if out:
        for line in out.strip().split("\n"):
            fields = line.split("|")
            if len(fields) < 3:
                continue
            jid_field = fields[0].strip()
            if "." not in jid_field:
                wall = fields[1].strip()
            elif jid_field.endswith(".batch"):
                rss = fields[2].strip()
    return {"job_id": job_id, "wall_time": wall, "peak_rss": rss}
