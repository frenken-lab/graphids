# Core: Callbacks

Lightning owns the training loop. graphids only ships callbacks that
encode policy Lightning's stock callbacks don't:

- `Sha256ModelCheckpoint` — `lightning.pytorch.callbacks.ModelCheckpoint`
  + a `<ckpt>.sha256` sidecar so [`graphids._fs.atomic_load`](../reference/write-paths.md)
  can verify bytes on read (GPFS truncates surprise us; sidecar is the
  load-time integrity check).

`graphids._mlflow.MLflowSystemMetricsCallback` records phase throughput and native
MLflow system metrics. `graphids._mlflow.MLflowSystemMetricsSummaryCallback`
summarizes native metric histories for offline diagnostic runs.

Lightning owns the training loop and callback lifecycle.

## `graphids.core.callbacks`

::: graphids.core.callbacks
