# CLI

The current experiment surface is **compose → launch → inspect**.
Compose is pure YAML/JSON validation; launch runs one typed
`ExperimentConfig` in-process; inspect reads the run manifest and
event log.
See [`chassis-invariants.md`](https://github.com/frenken-lab/graphids/blob/main/.claude/rules/chassis-invariants.md)
for the four architectural properties (drift resistance, MLflow as
state store, reproduction contract, render purity).

| Stage  | Command                  | Module                   |
|--------|--------------------------|--------------------------|
| config | `graphids exp config <yaml>`  | [`config`](#graphids.cli.config) |
| launch | `graphids exp launch <yaml>`  | [`launch`](#graphids.cli.launch) |
| status | `graphids exp status <run>`    | [`status`](#graphids.cli.status) |

There is no separate `ops` CLI surface. The run config itself carries
the stage and the stage-specific payload, so `fit` / `test` / `extract`
/ `analyze` all go through [`graphids.exp.runtime.launch_run`](runtime.md)
and the run manifest records the exact config that launched.

`app.py` owns the root Typer app + shared option types.
`__main__.py` imports each submodule to register commands.

## `graphids.cli`

::: graphids.cli
    options:
      members: true
      show_submodules: true
