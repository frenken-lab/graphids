# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct instantiation.
`render(jsonnet_path, tla) → dict` → `validate_config` (Pydantic) →
`graphids.orchestrate.build_run` (importlib class_path instantiation
with signature-filtered kwargs). PyTorch Lightning was removed in favor
of a custom `graphids.core.trainer.Trainer`. Analyzer CLI is pure Typer —
derives `model_type` from the checkpoint's self-describing `class_path`
and dispatches artifacts via `ARTIFACTS_BY_MODEL_TYPE` in
`core/analysis/schemas.py` (no jsonnet stage).

## Architecture

One route. `python -m graphids submit <preset.jsonnet>` (SLURM) or `python -m graphids fit` (in-process) →

1. `render(config_path, tla, set_overrides)` from `graphids.config.jsonnet`
   renders the merged dict. Every preset under `configs/ablations/` is a
   top-level function that computes its own `run_dir` via
   `std.native('paths.run_dir')(dataset, group, variant, seed)` —
   registered as a Python `native_callback` pointing at
   `graphids.config.paths.run_dir`. `run_root` flows in as
   `std.extVar('run_root')` (set once by `render` from `RUN_ROOT`),
   replacing the per-preset TLA default that used to drift.
2. `--set a.b.c=v` flags pass through `cli/app.py:dotted_to_nested` →
   `render(set_overrides=...)` → `std.extVar('overrides')` → applied
   via `std.mergePatch` at each ablation preset's apex. One mechanism;
   no Python in-place mutator, no jsonnet `apply_dotted` recursion.
3. `ResolvedConfig.from_rendered(rendered, stage_name=<basename>)`
   (`orchestrate.py`) runs `validate_config` (Pydantic — null list
   fields, monitor consistency, class_path namespacing, logger/callback
   wiring) and pulls `run_dir` / `ckpt_file` from
   `trainer.default_root_dir`.
4. `build(resolved)` (`orchestrate.py`) runs `build_run` which
   imports class_paths via `importlib`, applies `filter_kwargs` against
   each target's `__init__` signature, builds callbacks + logger, and
   returns an `InstantiatedRun(trainer, model, datamodule)`.
5. `train(artifacts, resolved, resume_from=...)` then `evaluate(...)`
   run fit/test and touch `.train_complete` / `.test_complete` markers.

Multi-stage runs use a *plan* — a jsonnet file declaring `{ nodes: [...] }`.
Shipped plans live under `configs/plans/`; `configs/plans/ofat.jsonnet`
is the OFAT topology. `python -m graphids run <plan.jsonnet> --dataset X
--seed N --cluster C` walks the plan in topological order via submitit,
threading each node's jid into downstream deps' `afterok`. FINISHED nodes
are skipped via an MLflow check before submission; `--force` overrides.
`python -m graphids status <plan.jsonnet> --dataset X --seed N` queries
MLflow per node and prints a status table. No bash manifest, no parser —
the plan jsonnet IS the artifact.

For atomic one-shot submissions (no plan), use `python -m graphids submit
<preset.jsonnet>`.

Full tree: `docs/reference/config-architecture.md`.

## File layout

```
configs/
├── _lib/defaults.libsonnet        # trainer / checkpoint / early_stopping defaults
├── ablations/{unsupervised,fusion,gat_sampling,gat_loss,id_encoding}/*.jsonnet
├── stages/{autoencoder,supervised,fusion}.jsonnet
├── models/
│   ├── {supervised,unsupervised,fusion}.libsonnet
│   └── fusion/{base,reward}.libsonnet + fusion/methods/*.libsonnet
├── plans/ofat.jsonnet             # multi-stage DAG topology
├── datasets/dataset_registry.json
├── matrix/{axes,topology}.json    # valid model types / stage existence
└── resources/submit_profiles.json # raw submitit kwargs, [mode][cluster][length]
```

`graphids/` package layout: see `ls graphids/` — every name is self-describing.
The non-obvious ones: `orchestrate.py` is a single module (not a subpackage)
holding `ResolvedConfig`, `InstantiatedRun`, `build_run`, `build`, `train`,
`evaluate`. `_mlflow.py` owns the entire MLflow surface (run lifecycle,
search filter, logged-model registration, dataset lineage).

## Running

```bash
# Local dev — renders defaults, trains to run_dir from jsonnet
python -m graphids fit --config configs/stages/autoencoder.jsonnet

# Override via TLA
python -m graphids fit \
    --tla 'dataset="hcrl_sa"' \
    --tla 'scale="large"' \
    --tla 'variational=false' \
    --config configs/stages/autoencoder.jsonnet \
    --set model.init_args.lr=0.005

# SLURM ablation
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
```

## Stage / ablation function convention

Every `stages/*.jsonnet` and `ablations/**/*.jsonnet` is a top-level
function with sensible defaults for every TLA. Adding a new TLA means
updating the jsonnet signature + (if the TLA is launcher-level) the
matching flat flag in `graphids/cli/submit.py` (which appends to the
inline `flag_tlas` list — there is no separate helper).

```jsonnet
// Stage (configs/stages/*.jsonnet) — no overrides TLAs.
function(
  dataset='hcrl_ch', seed=42, run_dir='',
  scale='small', conv_type='gatv2', variational=true,
  auxiliaries=[], vgae_ckpt_path=null,
  ckpt_path=null,
)
  defaults.trainer + defaults.checkpoint + defaults.early_stopping
  + vgae.base + vgae.scales[scale]
  + { seed_everything: seed, trainer+: {...}, data: {...}, model+: {...} }

// Ablation preset (configs/ablations/**/*.jsonnet) — wraps stage in mergePatch.
function(
  dataset=pd.dataset, seed=pd.seed,
  scale=pd.scale, conv_type=pd.conv_type,
  ckpt_path=null,
)
  std.mergePatch(
    stage(
      dataset=dataset, seed=seed, scale=scale,
      run_dir=std.native('paths.run_dir')(dataset, 'unsupervised', 'vgae', seed),
      conv_type=conv_type, model_type='vgae', variational=true,
      ckpt_path=ckpt_path,
    ) + { trainer+: { max_epochs: 1200 } },  // group defaults as nested obj
    std.extVar('overrides'),                 // user --set flags
  )
```

## Merge semantics

Jsonnet `+:` is deep-merge; `+` on top-level objects is shallow
merge-with-last-wins. Lists replace on conflict. Match the pattern from
existing stages religiously — a single missing `:` on a nested key
silently replaces the subtree instead of merging. Run
`~/.local/bin/jsonnet <path>.jsonnet` to verify a preset renders
correctly after editing.

## Robustness

1. **Typed TLA round-trip.** `render` passes every TLA through
   `--tla-code <k>=<json.dumps(v)>` so ints stay ints, bools stay bools,
   lists stay lists, `None` becomes jsonnet `null`.
2. **Pydantic gate (`ValidatedConfig`)** — null list fields in
   `model.init_args`, monitor mismatch between `checkpoint` and
   `early_stopping`, un-namespaced `class_path` strings, and
   `LearningRateMonitor` without `logger` all die with an actionable
   error before any torch import.
3. **Signature-filtered kwargs** — `build_run` runs every class_path's
   `init_args` through `filter_kwargs(klass, init_args)` so jsonnet can
   pass fields the target class doesn't accept without raising.
4. **`topology.py` import-time assertions** — every model family has a
   libsonnet, every stage has a `.jsonnet`, every fusion method has a
   method libsonnet; `submit_profiles.json` `scale_mult` keys are in
   `VALID_SCALES`. Missing files / bad keys fail at package import.

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
