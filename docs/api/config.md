# Config

Jsonnet → dict → Pydantic validation. The path through the system
is `render(jsonnet_path, tla) → dict` →
`BlueprintArray.model_validate(...)` →
`graphids.orchestrate.run_row` (importlib instantiation with
signature-filtered kwargs). See
[Config system](../reference/config-architecture.md) for the
runtime flow narrative.

- **`jsonnet`** — thin wrapper over the `_jsonnet` C bindings.
  TLAs are JSON-serialized so jsonnet receives real typed values
  (ints stay ints, bools stay bools, `None` becomes `null`). The
  binding is imported lazily inside `render` so this module stays
  safe to import on login nodes without `_jsonnet` installed.
- **`catalog`** — dataset catalog helpers and filesystem path
  primitives (`data_dir` / `cache_dir` under `LAKE_ROOT`,
  `load_catalog`, `dataset_names`).
- **`constants`** — project-wide constants and the typed view over
  `configs/matrix/axes.json`. Torchless; safe to import at package
  load.

`graphids.config.settings` (typed `pydantic-settings` view) and
`graphids.config.schemas` (Pydantic config schemas) were both
removed in the 2026-05-01 four-step rebuild — env vars are read
directly at the call sites that need them; the row schema lives in
[`graphids.blueprint`](orchestrate.md).

## `graphids.config`

::: graphids.config
    options:
      show_submodules: true
