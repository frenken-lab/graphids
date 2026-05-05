# Plan chassis

Python plan → list[dict] → Pydantic validation. The path through the
system is `plans.<name>.build(dataset, seed) → list[dict]` →
`Plan.model_validate(...)` →
`graphids.orchestrate.run_row` (importlib instantiation with
signature-filtered kwargs). See
[Config system](../reference/config-architecture.md) for the
runtime flow narrative.

The chassis is split across two top-level modules:

- **`graphids.paths`** — path math + dataset registry (`data_dir` /
  `cache_dir` under `$GRAPHIDS_LAKE_ROOT`, `run_dir` / `best_ckpt` /
  `states_dir` under `$GRAPHIDS_RUN_ROOT`, `load_catalog`,
  `dataset_names`, `PREPROCESSING_VERSION`, `CKPT_SUBPATH`,
  `LAST_CKPT_SUBPATH`, `PHASE_MARKERS`, `ModelType`). Torchless;
  import-safe from anywhere including login-node code paths.
- **`graphids.plan`** — composition + schema. Submodules:
    - `plan.lib` — class-path registry + spec helpers (`spec`,
      `can_bus`, `graph_dm`, `GAT`, `VGAE`, …).
    - `plan.compose` — `compose(...)` builder.
    - `plan.row` — `RowSpec` (composer-side mutable builder).
    - `plan.blueprint` — Pydantic `Row` discriminated union
      (`TrainRow`, `CacheRow`, `ExtractRow`, `AnalyzeRow`),
      `Plan`, `RenderedConfig`, `ClassPath`, `TrainerCfg`.
    - `plan.plans.{ablations,smoke,data}.<name>` — concrete plan
      modules exposing `build(*, dataset, seed) → list[dict]`.

`graphids.plan.settings` (typed `pydantic-settings` view) and
`graphids.plan.schemas` were removed in the 2026-05-01 four-step
rebuild — env vars are read directly at the call sites that need them;
the row + rendered-config schema lives in
[`graphids.plan.blueprint`](orchestrate.md).

## `graphids.paths`

::: graphids.paths
    options:
      show_submodules: true

## `graphids.plan`

::: graphids.plan
    options:
      show_submodules: true
