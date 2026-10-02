# GraphIDS Store Ownership

- `$GRAPHIDS_LAKE_ROOT` holds shared raw CSVs, versioned caches, and MLflow.
- `$GRAPHIDS_RUN_ROOT` holds per-user runs and checkpoints. On OSC it is
  `/fs/ess/PAS1266/graphids/dev/rf15`; diagnostic outputs belong under its
  `experimentruns/` directory, never the repository root.
- Normal runs use `{run_root}/{dataset}/{experiment_name}/{run_name}`.
- `$GRAPHIDS_SLURM_LOG_DIR` holds allocation scripts and stdout/stderr.

Temporal cache identity includes preprocessing version, representation,
vocabulary, training-source mode, and split policy. Do not change cache keys or
split manifests during structural refactors.

Workers write `.graphids/manifest.json`, `.graphids/events.jsonl`, and offline
`.graphids/mlflow_ingest.json`. `gx exp ingest` serializes MLflow writes after
training. Checkpoints stay under the run directory; preserve existing digest
sidecars and direct filesystem loading.

See `docs/reference/write-paths.md` and `docs/reference/data-architecture.md`.
