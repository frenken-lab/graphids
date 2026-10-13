# Core: Callbacks

Lightning owns the training loop. graphids only ships callbacks that
encode policy Lightning's stock callbacks don't:

- `Sha256ModelCheckpoint` — `lightning.pytorch.callbacks.ModelCheckpoint`
  + a `<ckpt>.sha256` sidecar so [`graphids._fs.atomic_load`](../reference/write-paths.md)
  can verify bytes on read (GPFS truncates surprise us; sidecar is the
  load-time integrity check).
- `SnapshotBatchProfilerCallback` — optional scored-bin, physical-time,
  event, node-bin, and graph-message instrumentation. Synchronization is opt-in.
- `ResourceUtilizationCallback` — MLflow system metric summaries.
- `VRAMDriftCallback` — device-memory drift checks between epochs.

Lightning owns the training loop and callback lifecycle.

## `graphids.core.callbacks`

::: graphids.core.callbacks
