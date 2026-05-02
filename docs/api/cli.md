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
| ops    | `graphids analyze` / `export` / `data` | [`analysis`](#graphids.cli.analysis), [`export`](#graphids.cli.export), [`data`](#graphids.cli.data) |

`app.py` owns the root Typer app + shared option types.
`__main__.py` imports each submodule to register commands.

## `graphids.cli`

::: graphids.cli
    options:
      members: true
      show_submodules: true
