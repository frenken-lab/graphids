# Run Execution

The experiment launcher is `graphids.exp.run`. It accepts a typed
`RunConfig`, constructs Ray `TorchTrainer` directly, and runs the worker loop
that builds data/model/trainer objects from YAML config.

Current stages:

- `fit` / `test` -> Lightning trainer launch with config-driven data/model
  instantiation, Ray Lightning strategy/environment, and Ray checkpoint/metric
  reporting.

Ray normally uses its report callback for checkpointing. Supplying a Lightning
`ModelCheckpoint` callback also enables Lightning checkpointing by default; an
explicit `trainer.enable_checkpointing` setting takes precedence.

`Sha256ModelCheckpoint(save_last=True)` saves the latest eligible epoch even
when its monitored score ties or regresses. Best-checkpoint selection remains
separate, and every saved checkpoint has a SHA256 sidecar.

## `graphids.exp.run`

::: graphids.exp.run

`execute_run` also supports local Lightning diagnostics. `records` owns durable
status/results; `tracking` owns MLflow exports. Execution is limited to one
node/device, matching the active event contract.
