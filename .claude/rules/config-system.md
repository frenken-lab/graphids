# GraphIDS Config System

Python-native composition + Pydantic validation + direct instantiation.
`build(dataset, seed)` (a Python plan under
`graphids/plan/plans/`) → `list[dict]` → `Plan.model_validate(...)`
→ `graphids.orchestrate.run_row` (importlib `class_path` instantiation
with signature-filtered kwargs).

The jsonnet layer (`gojsonnet` + `configs/*.libsonnet`) was deleted
2026-05-04. Don't reach for it. New plans are Python modules.

> Architecture detail, file layout, examples:
> `docs/reference/config-architecture.md`. This file owns the rules
> that bite during edits.

## Composition rules

- **Plans live at `graphids/plan/plans/<name>.py`** and expose
  `def build(*, dataset: str, seed: int) -> list[dict]`.
  `graphids run <name>` imports and calls.
- **Composer** (`graphids/plan/compose.py`) returns a frozen
  :class:`graphids.plan.compose.RowSpec`. Its `rendered` field is a
  frozen Pydantic :class:`graphids.plan.schema.RenderedConfig` — typo'd
  field access raises ``AttributeError``; constructing one with an
  unknown key raises ``pydantic.ValidationError``. Plan code must not
  mutate the rendered spec; build a new RowSpec or pass
  `trainer_overrides=`.
- **Primitives** (`graphids/plan/primitives.py`) return bare
  ``{class_path, init_args}`` dicts. The composer merges, validates,
  and emits.
- **Loss fragments** are `{"loss_fn": {class_path, init_args}}` blocks
  that the composer `update`s into `model.init_args` — no deep-merge
  magic, one literal call site (`compose.py::compose`).
- **Plan authors import from `graphids.plan`** (the package's public
  surface re-exports `compose`, `fusion`, `extract`, primitives, and
  class-path constants). Internal modules (`schema`, `compose`,
  `primitives`) are an implementation detail.

## Path layout

Path math lives in `graphids/paths.py` (`run_dir`, `best_ckpt`,
`states_dir`, `load_catalog`). Plans and the composer call it directly;
no native-callback bridge.

```
{RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`GRAPHIDS_RUN_ROOT` is required (no default — fail-fast in `_run_root()`).

## Null preservation

Python `None` is a real value (e.g. `gradient_clip_val: None` for
fusion's RL methods). Pydantic round-trips it as JSON `null` through the
typed `RenderedConfig` (`graphids/plan/schema.py`). Don't replace
with sentinels.

## Environment variables

Read directly from `os.environ` at call sites (`graphids/orchestrate.py`,
`graphids/slurm/submit.py`, `graphids/_mlflow.py`). The old typed
`GraphIDSSettings` was deleted in the 2026-05-01 four-step rebuild —
pydantic-settings paid for nothing once the surface shrank. Path roots
(`LAKE_ROOT` vs `RUN_ROOT`): see `data-layout.md`.

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
