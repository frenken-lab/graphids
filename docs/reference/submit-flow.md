# Submit Flow

> Status: **historical archive** | Companion: `runtime.md`

This page preserves the old SLURM submission story for reference. The
live experiment path now goes through `graphids.exp.runtime.launch_run`
and `graphids exp launch <experiment.yaml>`.

The legacy model was row-based and plan-driven; the current model is
typed and launch-driven. The architectural guarantee that remains is the
same: the compute node re-imports source at execution time, so code
committed after submission is still visible to the job.

## Historical shape

- Render a plan.
- Submit one job per rendered unit.
- Re-import source on the compute node.
- Resume preemption by letting Lightning requeue the current job.

## Current shape

1. Validate an `ExperimentConfig`.
2. Launch it through `graphids.exp.runtime.launch_run`.
3. Write a manifest and event log for the run.

## Files of interest

- `graphids/cli/exp.py`
- `graphids/exp/config.py`
- `graphids/exp/runtime.py`
- `graphids/slurm/submit.py`
