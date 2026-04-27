# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct instantiation.
`render(jsonnet_path, tla) → dict` → `validate_config` (Pydantic) →
`graphids.orchestrate.build_run` (importlib class_path instantiation
with signature-filtered kwargs).

> Architecture detail, file layout, stage convention, robustness behaviors,
> running examples: `docs/reference/config-architecture.md`.

This file owns the rules that bite during edits: merge semantics,
null preservation, env vars, path scheme, observability wiring.

## Merge semantics

Jsonnet `+:` is deep-merge; `+` on top-level objects is shallow
merge-with-last-wins. Lists replace on conflict. Match the pattern from
existing stages religiously — a single missing `:` on a nested key
silently replaces the subtree instead of merging. Run
`~/.local/bin/jsonnet <path>.jsonnet` to verify a preset renders
correctly after editing.

`--set a.b.c=v` flags pass through `cli/app.py:dotted_to_nested` →
`render(set_overrides=...)` → `std.extVar('overrides')` → applied via
`std.mergePatch` at each ablation preset's apex. One mechanism — no
Python in-place mutator, no jsonnet `apply_dotted` recursion.

## Null preservation

`data.init_args.num_workers: null` is a real value (auto-sized from
GPU-first sizing), not "missing". Jsonnet has a first-class `null` —
preserve it. The autoencoder stage emits `num_workers: null`
explicitly; `supervised.libsonnet` overrides it to `4` because GAT is
compute-bound.

## Environment variables

Typed in `GraphIDSSettings` (`config/settings.py`); pydantic-settings
auto-loads `./.env` from the project root. `extra="ignore"` so shell-only
`GRAPHIDS_*` vars (read by `_preamble.sh` etc.) don't trip validation.
Path roots (`LAKE_ROOT` vs `RUN_ROOT`): see `data-layout.md`.

## Path layout

Path scheme lives in **`graphids/config/paths.py`** (Python) and is
exposed to jsonnet via `native_callbacks` in `render()` —
`std.native('paths.run_dir')(dataset, group, variant, seed)` etc. Both
sides call the same Python source of truth, no parallel jsonnet impl.

```
{RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`run_root` is required (no default — fail-fast). `slurm/dag.py`
imports `from graphids.config import paths` and uses the same module;
no separate `_run_dir` math.

## Observability (MLflow + OpenTelemetry)

Storage layout + store-ownership table: `data-layout.md`. This file owns
the wiring details:

- **Lifecycle wiring**: `_mlflow.start_training_run` opens the fit run
  inside `orchestrate.train` before `trainer.fit`; `MLflowTrainingCallback`
  (`core/mlflow_callback.py`) forwards `callback_metrics` per epoch and
  closes the run in `on_fit_end`. Test phase opens its own always-fresh
  run via `_mlflow.log_test_run`. Experiment is per-axis: `graphids/{dataset}/{group}`.
- **Resume gating** (fit only): status-gated on matching `run_name` +
  `phase=fit` (FAILED/KILLED → resume; RUNNING/FINISHED refuse unless
  `GRAPHIDS_FORCE_RESUME=1`; git-SHA change → new run).
- **Failure mode**: MLflow is a hard dep, exceptions propagate. Two
  documented soft-failures: `MlflowException` on resume `log_params`
  conflict, and `end_training_run` cleanup (logged-not-raised so secondary
  failures don't shadow training exceptions via `__context__`).
- **Query API**: always go through `_mlflow.build_search_filter(...)`.
  Hand-composed `filter_string=` strings drift across callers (dataset,
  group, variant, seed, phase, cluster, status all need consistent quoting).
- **OTel `traces.jsonl`** (per-`run_dir`): single `training.fit` span +
  structured-log events. Single-run debugging only — cross-run analysis
  is MLflow's job.
