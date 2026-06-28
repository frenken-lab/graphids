# Ray Launcher

The experiment launcher is `graphids.exp.ray_backend`. It accepts a typed
`RunConfig`, constructs Ray `TorchTrainer` directly, and runs the worker loop
that builds data/model/trainer objects from YAML config.

Current stages:

- `fit` / `test` -> Lightning trainer launch with config-driven data/model
  instantiation, Ray Lightning strategy/environment, and Ray checkpoint/metric
  reporting.

## `graphids.exp.ray_backend`

::: graphids.exp.ray_backend
