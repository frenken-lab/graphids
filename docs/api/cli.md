# CLI

The four-step chassis: **render → blueprint → exec → submit**. Each
command does exactly one thing and feeds the next; no stage submits,
queries MLflow across runs, or orchestrates multiple jobs. See
[`single-submission-primitive.md`](https://github.com/frenken-lab/graphids/blob/main/.claude/rules/single-submission-primitive.md)
for the architectural rule.

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
