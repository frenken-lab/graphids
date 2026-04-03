# Orchestrator: no duplicate job detection

## Problem

Restarting the orchestrator while child SLURM jobs are running causes
duplicate submissions. The orchestrator only checks for checkpoint files
on disk — it has no mechanism to detect that a SLURM job is already
running for the same asset.

## Impact

Wasted GPU hours from duplicate training runs. Requires manual
`scancel` of old jobs after restarting the orchestrator.

## Possible fixes

1. **Lock file per asset** — write `{run_dir}/.running_{job_id}` at
   submission, check at start, clear on completion/failure
2. **sacct query** — before submitting, check `sacct` for running jobs
   with matching job name pattern
3. **SLURM job dependency** — track child job IDs in orchestrator state,
   check `squeue` before resubmit

Option 2 is simplest (~10 lines in `submit_and_wait`).

## Also: turm only shows stdout

turm tails SLURM stdout but training output (Lightning, structlog) goes
to stderr. No live visibility into training progress via turm. Would need
turm to tail both `.out` and `.err` files, or redirect stderr to stdout
in the SLURM script (`2>&1`).
