# Experiment execution invariants

The active path is experiment YAML → typed run → Ray/SLURM → Lightning.
The former Jsonnet plan/row chassis is archived with the snapshot implementation.

- Validate and inspect configuration before launch. `gx exp config` and
  `gx exp submit --dry-run` are read-only inspection paths.
- Submission saves expanded YAML so queued jobs do not re-expand changed bases.
  Python source still comes from the selected checkout; freeze source for comparisons.
- Workers journal run status and offline metrics under `.graphids/`.
  `gx exp ingest` serializes writes to the shared MLflow store.
- Preserve experiment identity, source provenance, output paths, and checkpoint
  compatibility during structural refactors.

See `config-system.md`, `slurm-hpc.md`, and `docs/reference/submit-flow.md`.
