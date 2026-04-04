"""Finalize run_record.json after test+analyze phases complete.

Called from the generated sbatch script (after test/analyze, before _epilog.sh).
Updates the sidecar with phase marker status and SLURM wall time from sacct.

Usage:
    python -m graphids _finalize-record --run-dir /path/to/run_dir
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from graphids.config import PHASE_MARKERS
from graphids.core.contracts.run_record import read_run_record, write_run_record


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Finalize run_record.json with phases + sacct")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    record = read_run_record(args.run_dir)
    if record is None:
        return  # no sidecar to finalize (legacy run or callback disabled)

    # Phase markers
    phases = {
        phase: (args.run_dir / marker).exists()
        for phase, marker in PHASE_MARKERS.items()
    }

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

    record = record.model_copy(update={
        "phases": phases,
        **({"wall_time_seconds": wall_time_seconds} if wall_time_seconds is not None else {}),
    })
    write_run_record(record, args.run_dir)
