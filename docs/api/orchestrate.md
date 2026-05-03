# Orchestrate

Dict→objects→`pl.Trainer.fit` bridge. Walks `row.rendered_config`,
instantiates each `{class_path, init_args}` block via `importlib`,
opens/closes the MLflow run, and dispatches on `row.action`. The
training loop itself is `lightning.pytorch.Trainer`; graphids-specific
callbacks ship in [`graphids.core.callbacks`](trainer.md).

graphids hooks before Lightning takes over: `dm.bind(model, device)`
→ `dm.setup(stage)` → `model.prepare_from_datamodule(dm)` (lazy
`_build()` with DM-resolved `num_ids`/`in_channels`/`num_classes`)
→ `pl.Trainer(...).fit(model, train_dataloaders=..., val_dataloaders=...)`.
The DM is stashed at `pl_module._graphids_dm` so callbacks can find
it (Lightning's `trainer.datamodule` is None when fit gets dataloaders
directly, which is the path graphids uses since DMs aren't
`pl.LightningDataModule` subclasses).

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
