# SLURM: Sizing

Optional walltime estimation from MLflow run history. ``python -m graphids
submit --time-from-history`` calls ``estimate_walltime_minutes`` to tighten
the wall limit for ``(cluster, group, dataset)`` combinations with ≥3
prior FINISHED runs; ``None`` means fall back to the static per-length
default in ``configs/resources/submit_profiles.json``.

Formula: ``ceil(p95(elapsed_mins) × 1.5)`` clamped to ``[10, 7 days]``.

## `graphids.slurm.sizing`

::: graphids.slurm.sizing
