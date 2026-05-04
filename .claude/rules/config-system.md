# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct instantiation.
`render(jsonnet_path, tla) → dict` → `validate_config` (Pydantic) →
`graphids.orchestrate.build_run` (importlib class_path instantiation
with signature-filtered kwargs).

> Architecture detail, file layout, examples:
> `docs/reference/config-architecture.md`. This file owns the rules
> that bite during edits.

## Merge semantics

Jsonnet `+:` is deep-merge; `+` on top-level objects is shallow
merge-with-last-wins. Lists replace on conflict. Match the pattern from
existing stages — a single missing `:` on a nested key silently replaces
the subtree instead of merging. Run `~/.local/bin/jsonnet <path>.jsonnet`
to verify after editing.

`--set a.b.c=v` flags pass through `cli/app.py:dotted_to_nested` →
`render(set_overrides=...)` → `std.extVar('overrides')` → applied via
`std.mergePatch` at each ablation preset's apex. One mechanism — no
Python in-place mutator, no jsonnet recursion.

## Null preservation

`data.init_args.num_workers: null` is a real value (auto-sized from
GPU-first sizing), not "missing". Jsonnet has first-class `null` —
preserve it. The autoencoder stage emits `num_workers: null`;
`supervised.libsonnet` overrides to `4` because GAT is compute-bound.

## Environment variables

Read directly from `os.environ` at call sites (`graphids/runtime.py`,
`graphids/slurm/submit.py`, `graphids/_mlflow.py`). The old typed
`GraphIDSSettings` was deleted in the 2026-05-01 four-step rebuild —
pydantic-settings paid for nothing once the surface shrank. Path roots
(`LAKE_ROOT` vs `RUN_ROOT`): see `data-layout.md`.

## Path layout

Path scheme is computed inside the plan jsonnets using `run_root` +
`dataset` + `group` + `variant` + `seed` TLAs. There is no Python
`config/paths.py` and no `native_callbacks` shim — jsonnet owns path
math, and the rendered `run_dir` flows through `TrainRow` to every
consumer (orchestrate, MLflow, ckpt I/O).

```
{RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`run_root` is required (no default — fail-fast).

## Observability wiring

Storage + store-ownership: `data-layout.md`. Wiring rules:

- **Lifecycle**: `_mlflow.start_training_run` opens the fit run inside
  `orchestrate.run_row` before `trainer.fit`; `MLflowTrainingCallback`
  emits one `log_batch` per epoch and closes in `on_fit_end`. Test phase
  opens its own always-fresh run. Experiment: `graphids/{dataset}/{group}`.
- **Resume gating** (fit only): FAILED/KILLED → resume same `run_id`;
  RUNNING/FINISHED refuse unless `GRAPHIDS_FORCE_RESUME=1`; git-SHA change → new run.
- **Failure mode**: MLflow is a hard dep, exceptions propagate.
- **Query API**: always `_mlflow.build_search_filter(...)`.
- **No OTel.** Single sink: stderr → SLURM `*_log.err`.
