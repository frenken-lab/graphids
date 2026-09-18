# Artifacts

> Status: **historical**

The old `graphids.core.artifacts` package was removed during the temporal
training-surface cleanup. Current runs write GraphIDS journals under
`.graphids/`, Ray checkpoints when enabled, and optional MLflow ingest payloads.

Use these current references instead:

- [`graphids.exp.ray_backend`](ray-backend.md) for launch-time result reporting.
- [`graphids.exp.ingest`](../reference/observability.md) for offline MLflow
  serialization.
- [`docs/reference/write-paths.md`](../reference/write-paths.md) for run,
  checkpoint, artifact, and MLflow paths.

