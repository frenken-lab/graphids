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

Read directly from `os.environ` at the call sites that need them
(`graphids/runtime.py`, `graphids/slurm/submit.py`, `graphids/_mlflow.py`).
The old typed `GraphIDSSettings` / `config/settings.py` was deleted in
the 2026-05-01 four-step rebuild — pydantic-settings was paying for
nothing once the surface shrank. Path roots (`LAKE_ROOT` vs `RUN_ROOT`):
see `data-layout.md`.

## Path layout

Path scheme is computed inside the plan jsonnets themselves using
`run_root` + `dataset` + `group` + `variant` + `seed` TLAs. There is no
longer a Python `graphids/config/paths.py` (deleted 2026-05-01) and no
`native_callbacks` shim — jsonnet owns path math, and the rendered
`run_dir` flows through the validated `TrainRow` to every consumer
(orchestrate, MLflow, ckpt I/O). Single source of truth, single
direction.

```
{RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`run_root` is required (no default — fail-fast).

## Observability (MLflow + structlog)

Storage layout + store-ownership table: `data-layout.md`. This file owns
the wiring details:

- **Lifecycle wiring**: `_mlflow.start_training_run` opens the fit run
  inside `orchestrate.run_row` before `trainer.fit`;
  `MLflowTrainingCallback` (`graphids/_mlflow.py`) emits one `log_batch`
  per epoch via the sanitized-metric path and closes the run in
  `on_fit_end`. Test phase opens its own always-fresh run via
  `_mlflow.log_test_run`. Experiment is per-axis:
  `graphids/{dataset}/{group}`.
- **Resume gating** (fit only): status-gated on matching `run_name` +
  `phase=fit` (FAILED/KILLED → resume; RUNNING/FINISHED refuse unless
  `GRAPHIDS_FORCE_RESUME=1`; git-SHA change → new run). Real-sqlite
  resume matrix is exercised by tests in `tests/test_mlflow.py`.
- **Failure mode**: MLflow is a hard dep, exceptions propagate. Two
  documented soft-failures: `MlflowException` on resume `log_params`
  conflict, and `end_training_run` cleanup (logged-not-raised so
  secondary failures don't shadow training exceptions via `__context__`).
- **Query API**: always go through `_mlflow.build_search_filter(...)`.
  Hand-composed `filter_string=` strings drift across callers (dataset,
  group, variant, seed, phase, cluster, status all need consistent
  quoting).
- **No OTel.** `_otel.py` was deleted 2026-05-01 — it was structlog
  config under a misleading name (spans were never opened, `traces.jsonl`
  was always empty). The salvageable parts (structlog → JSON stderr,
  SLURM-context auto-injection) inlined into
  `graphids/runtime.py:_configure_logging`. Single sink: stderr → SLURM
  `*_log.err`. Cross-run analysis is MLflow's job; single-run debugging
  is `slurm_logs/<jid>.err`.
