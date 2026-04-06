"""SLURM accounting helpers (sacct parsing)."""

from __future__ import annotations

import os
import subprocess

from graphids.log import get_logger

log = get_logger(__name__)


def sacct_query(
    job_ids: list[str] | list[int], fmt: str, *, units: str = "G", cluster: str | None = None
) -> str:
    """Run sacct and return stdout. Shared by poll() and profiler."""
    ids = ",".join(str(j) for j in job_ids)
    cmd = ["sacct", "-j", ids, "--parsable2", "--noheader", f"--format={fmt}", f"--units={units}"]
    if cluster:
        cmd.extend(["-M", cluster])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("sacct_error", stderr=r.stderr.strip())
        return ""
    return r.stdout


def sacct_by_user(
    fmt: str = "JobIDRaw,JobName,State,Elapsed", *, starttime: str = "now-30days"
) -> str:
    """Run sacct for current user's recent jobs. Returns parsable stdout."""
    user = os.environ.get("USER", "")
    if not user:
        return ""
    cmd = [
        "sacct",
        "-u",
        user,
        f"--starttime={starttime}",
        "--parsable2",
        "--noheader",
        f"--format={fmt}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        log.warning("sacct_error", stderr=r.stderr.strip())
        return ""
    return r.stdout


def parse_elapsed(s: str) -> float | None:
    """Parse sacct Elapsed (``HH:MM:SS`` or ``D-HH:MM:SS``) to seconds.

    Returns ``None`` on unparseable input (empty string, malformed fields).
    """
    if not s:
        return None
    try:
        parts = s.replace("-", ":").split(":")
        nums = [float(p) for p in parts]
        return sum(v * m for v, m in zip(reversed(nums), (1, 60, 3600, 86400)))
    except ValueError:
        return None


def job_accounting(job_id: int) -> dict[str, str | int]:
    """Return ``{job_id, wall_time, peak_rss}`` parsed from sacct.

    sacct emits multiple rows per job — the parent row carries Elapsed, the
    ``.batch`` child row carries MaxRSS. This walks both and returns the
    merged postmortem used by dagster asset metadata.
    """
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
