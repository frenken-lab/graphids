# Core: Callbacks

Lightning owns the training loop. graphids only ships callbacks that
encode policy Lightning's stock callbacks don't:

- `Sha256ModelCheckpoint` — `lightning.pytorch.callbacks.ModelCheckpoint`
  + a `<ckpt>.sha256` sidecar so [`graphids._fs.atomic_load`](../reference/write-paths.md)
  can verify bytes on read (GPFS truncates surprise us; sidecar is the
  load-time integrity check).
- `TauNormCallback` — Kang ICLR 2020 τ-norm of the GAT classifier head
  at fit-end. Loads from the best ckpt, in-place rescales the final
  `fc_layers[-1]` `nn.Linear` weight by `‖w_c‖^τ`, re-saves.
- `VRAMDriftCallback` — warns once when free VRAM shrinks past
  `threshold` between epochs (probe baseline at fit-start).

`pl.callbacks.EarlyStopping` is wired straight from the libsonnet —
graphids no longer ships its own.

`MLflowTrainingCallback` (in [`graphids._mlflow`](runtime.md))
forwards per-epoch metrics + run-config + LoggedModel registration; it
is registered alongside but lives in the MLflow surface for
discoverability.

The training loop, AMP autocast, gradient clipping, optimizer state,
scheduler stepping, ckpt save/load schema, and the callback lifecycle
all live in `lightning.pytorch.Trainer`. The `core/trainer.py`,
`core/_metric_acc.py`, and `core/_ckpt.py` modules that previously
re-implemented these were removed in the 2026-05-02 Lightning migration
(commit `c974185`); see `~/plans/lightning-migration-spike.md` for the
inventory of what migrated and what was kept.

## `graphids.core.callbacks`

::: graphids.core.callbacks
