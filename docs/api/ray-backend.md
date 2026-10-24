# Ray Launcher

The experiment launcher is `graphids.exp.ray_backend`. It accepts a typed
`RunConfig`, constructs Ray `TorchTrainer` directly, and runs the worker loop
that builds data/model/trainer objects from YAML config.

Current stages:

- `fit` / `test` -> Lightning trainer launch with config-driven data/model
  instantiation, Ray Lightning strategy/environment, and Ray checkpoint/metric
  reporting.

Ray normally uses its report callback for checkpointing. Supplying a Lightning
`ModelCheckpoint` callback also enables Lightning checkpointing by default; an
explicit `trainer.enable_checkpointing` setting takes precedence.

## `graphids.exp.ray_backend`

::: graphids.exp.ray_backend
