# Orchestrate

Dict→objects→`Trainer.fit` bridge. Walks `row.rendered_config`,
instantiates each `{class_path, init_args}` block via `importlib`,
opens/closes the MLflow run, and dispatches on `row.action`. The
training loop itself lives in [`graphids.core.trainer`](trainer.md).

`MLflowTrainingCallback` reads its `run_id` from
`$GRAPHIDS_MLFLOW_RUN_ID` (set by `train` / `evaluate` immediately
after `start_training_run`), so the libsonnet's `init_args: {}` is
honest — every callback now goes through `_instantiate` uniformly.

## `graphids.orchestrate`

::: graphids.orchestrate

## `graphids.blueprint`

`TrainRow` / `ExtractRow` / `BlueprintArray` — the validated row
schema produced by `graphids run` and consumed by `graphids exec` /
`graphids submit`. Single source of truth for what a stage needs.

::: graphids.blueprint
