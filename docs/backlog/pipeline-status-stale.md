# pipeline-status shows stale dagster state, not SLURM reality

> Created: 2026-04-02 | Priority: HIGH

## Problem

`python -m graphids pipeline-status` showed only 2 SUCCESS assets when sacct
confirmed 14 completed (exit 0) + 5 fusions. The tool queries dagster's
`AssetRecord` via `get_asset_records()`, which tracks dagster run status —
not SLURM job exit codes.

Dagster marks a run as STARTED when `sbatch` is submitted but never updates
to SUCCESS when the SLURM job completes. The `SlurmTrainingComponent` uses
`sacct` polling in `execution.py` to detect completion, but this status is
not propagated back to the dagster `AssetRecord`.

## Root cause

The dagster `@asset` function in `assets.py` returns (materializes) when
the SLURM job finishes. But if the orchestrator CPU job dies, gets OOM'd,
or simply hasn't polled yet, the dagster run stays STARTED forever.

The phase markers (T/E/A columns all showing `-`) confirm: dagster never
read the `.train_complete` / `.test_complete` markers because it never
re-evaluated the asset after SLURM completion.

## Impact

Any operator relying on `pipeline-status` to assess ablation progress gets
a misleading picture. The 2 SUCCESS entries were from the hcrl_sa smoke test
(different dagster run), not the current ablation.

## Fix options

**A. Reconcile from sacct.** `pipeline-status` cross-references dagster
records with `sacct` output. If sacct says COMPLETED and dagster says
STARTED, show COMPLETED. ~20 lines in `commands/pipeline_status.py`.

**B. Fix the source.** Ensure `execution.py` polling loop updates dagster
run status on SLURM completion. Requires dagster instance API call.

**Recommendation:** A first (quick, fixes the reporting), B later (fixes
the root cause).
