# Orchestrate: Config

Boundary types shared by the Typer CLI and the stage primitives.
``ResolvedConfig`` is the handoff from ``render`` +
[`validate_config`](config.md) into
[`build`](orchestrate_stage.md) / [`train`](orchestrate_stage.md) /
[`evaluate`](orchestrate_stage.md); ``InstantiatedRun`` is the wired
``(trainer, model, datamodule)`` triple produced by
[`build_run`](orchestrate_instantiate.md).

## `graphids.orchestrate.config`

::: graphids.orchestrate.config
