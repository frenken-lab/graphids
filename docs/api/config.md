# Config

Python plan → list[dict] → Pydantic validation. The path through the
system is `plans.<name>.build(dataset, seed) → list[dict]` →
`BlueprintArray.model_validate(...)` →
`graphids.orchestrate.run_row` (importlib instantiation with
signature-filtered kwargs). See
[Config system](../reference/config-architecture.md) for the
runtime flow narrative.

`graphids.config` is the legacy package that owns path/dataset
catalog helpers (`run_dir`, `best_ckpt`, `load_catalog`, `data_dir`,
`cache_dir`, `states_dir`). The composition layer
(`graphids.configs.*`) imports from here directly.

- **`catalog`** — dataset catalog helpers and filesystem path
  primitives (`data_dir` / `cache_dir` under `LAKE_ROOT`,
  `run_dir` / `best_ckpt` / `states_dir` under `RUN_ROOT`,
  `load_catalog`, `dataset_names`).
- **`constants`** — project-wide constants and the typed view over
  `configs/matrix/axes.json`. Torchless; safe to import at package
  load.

`graphids.config.settings` (typed `pydantic-settings` view) and
`graphids.config.schemas` were removed in the 2026-05-01 four-step
rebuild — env vars are read directly at the call sites that need
them; the row + rendered-config schema lives in
[`graphids.configs.blueprint`](orchestrate.md).

## `graphids.config`

::: graphids.config
    options:
      show_submodules: true
