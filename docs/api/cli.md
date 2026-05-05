# CLI

The chassis: **render → exec / submit**. Render is pure JSON;
exec runs one row in-process; submit either takes one row or, via
``plans submit``, walks a rendered plan with MLflow-aware filtering.
See [`chassis-invariants.md`](https://github.com/frenken-lab/graphids/blob/main/.claude/rules/chassis-invariants.md)
for the four architectural properties (drift resistance, MLflow as
state store, reproduction contract, render purity).

| Stage  | Command                          | Module                     |
|--------|----------------------------------|----------------------------|
| render | `graphids run <plan>`            | [`run`](#graphids.cli.run) |
| exec   | `graphids exec --row <json>`     | [`exec`](#graphids.cli.exec) |
| submit | `graphids submit --row <json>`   | [`submit`](#graphids.cli.submit) |

There is no separate `ops` CLI surface. Ops jobs (per-checkpoint
artifacts, fusion-feature extraction, ad-hoc shell) are blueprint
actions — `analyze` / `extract` / `cmd` rows authored in a plan
jsonnet and run/submitted through the same three commands.

`exec` dispatches on `row.action` via
[`graphids.orchestrate.run_row`](orchestrate.md#graphids.orchestrate.run_row):
`fit` / `test` → train+eval; `extract` →
fusion-feature cache build; `analyze` →
[`graphids.core.artifacts.Analyzer`](artifacts.md).

`app.py` owns the root Typer app + shared option types.
`__main__.py` imports each submodule to register commands.

## `graphids.cli`

::: graphids.cli
    options:
      members: true
      show_submodules: true
