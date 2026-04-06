"""Finalize run_record.json after test+analyze phases complete.

Operation layer — argparse surface lives in
``graphids.commands.finalize_record``. Called from the generated sbatch
script (after test/analyze, before _epilog.sh). Updates the sidecar with
phase marker status and SLURM wall time from sacct.
"""

from __future__ import annotations

import os
from pathlib import Path

from graphids.config.constants import PHASE_MARKERS
from graphids.core.io import read_run_record, write_run_record


def finalize_run_record(run_dir: Path) -> None:
    """Update ``run_dir/run_record.json`` with phase markers + sacct wall time.

    No-op when the sidecar doesn't exist (legacy run or callback disabled).
    """
    record = read_run_record(run_dir)
    if record is None:
        return

    # Phase markers
    phases = {phase: (run_dir / marker).exists() for phase, marker in PHASE_MARKERS.items()}

    # SLURM wall time from sacct
    wall_time_seconds = None
    job_id_str = os.environ.get("SLURM_JOB_ID")
    if job_id_str:
        from graphids.slurm import parse_elapsed, sacct_query

        out = sacct_query([int(job_id_str)], "Elapsed")
        if out:
            for line in out.strip().splitlines():
                fields = line.split("|")
                if fields and "." not in fields[0].strip():
                    wall_time_seconds = parse_elapsed(
                        fields[-1].strip() if len(fields) > 1 else fields[0].strip()
                    )
                    break

    record = record.model_copy(
        update={
            "phases": phases,
            **({"wall_time_seconds": wall_time_seconds} if wall_time_seconds is not None else {}),
        }
    )
    write_run_record(record, run_dir)
