# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct instantiation.
`render(jsonnet_path, tla) → dict` → `validate_config` (Pydantic) →
`graphids.orchestrate.instantiate.build_run` (importlib class_path instantiation
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
   (`orchestrate/config.py`) runs `validate_config` (Pydantic — null list
   fields, monitor consistency, class_path namespacing, logger/callback
   wiring) and pulls `run_dir` / `ckpt_file` from
   `trainer.default_root_dir`.
4. `build(resolved)` (`orchestrate/stage.py`) runs `build_run` which
   imports class_paths via `importlib`, applies `filter_kwargs` against
   each target's `__init__` signature, builds callbacks + logger, and
   returns an `InstantiatedRun(trainer, model, datamodule)`.
5. `train(artifacts, resolved, resume_from=...)` then `evaluate(...)`
   run fit/test and touch `.train_complete` / `.test_complete` markers.

Multi-stage chains (e.g. the KD chain autoencoder → supervised →
fusion) are bash loops that submit each preset with
`SBATCH_DEP=afterok:<jid>` — see `graphids.slurm.dag` (CLI: `python -m graphids launch-ablation`).
No in-process pipeline driver, no planner, no identity-hash layer.

Full tree: `docs/reference/config-architecture.md`.

## File layout

```
configs/
├── _lib/
│   └── defaults.libsonnet         # trainer / checkpoint / early_stopping defaults
├── ablations/
│   ├── unsupervised/{vgae,gae,dgi}.jsonnet
│   ├── fusion/{bandit,dqn,mlp,weighted_avg}.jsonnet
│   ├── conv_type/ gat_sampling/ gat_loss/
│   └── README.md
├── datasets/
│   └── dataset_registry.json      # dataset catalog (domain → dataset metadata)
├── matrix/
│   ├── axes.json                  # valid model types / scales / fusion methods
│   └── topology.json              # stage name list (existence check only)
├── stages/
│   ├── autoencoder.jsonnet
│   ├── supervised.jsonnet
│   └── fusion.jsonnet
├── models/
│   ├── unsupervised.libsonnet     # { base, scales, kd }
│   └── supervised.libsonnet
├── resources/
│   └── submit_profiles.json       # two entries: gpu, cpu (per-cluster partitions + per-length wall defaults)
├── fusion.libsonnet               # { base, methods: {bandit, dqn, mlp, weighted_avg} }
└── fusion/
    ├── base.libsonnet
    └── methods/{bandit,dqn,mlp,weighted_avg}.libsonnet

graphids/
  cli/
    app.py                         # Typer root app + shared option types + dotted_to_nested
    training.py                    # fit / test commands
    analysis.py                    # analyze command
    data.py                        # rebuild-caches, extract-fusion-states
  orchestrate/
    __init__.py                    # re-exports
    config.py                      # ResolvedConfig, InstantiatedRun
    instantiate.py                 # build_run (trainer, model, datamodule)
    stage.py                       # build, train, evaluate
  config/
    __init__.py                    # public API facade
    constants.py                   # CONFIG_DIR, PROJECT_ROOT, LAKE_ROOT, RUN_ROOT
    settings.py                    # GraphIDSSettings — pydantic-settings auto-loads ./.env
    paths.py                       # run_dir / vgae_ckpt / states_dir — canonical scheme
    topology.py                    # stage-file existence check, dataset catalog
    schemas.py                     # validate_config → ValidatedConfig (Pydantic)
    jsonnet.py                     # render(path, tla, set_overrides) — registers native_callbacks
  core/
    trainer.py                     # Trainer, TrainerConfig, seed_everything
    callbacks.py                   # CallbackBase, ModelCheckpoint, EarlyStopping, VRAMDriftCallback
    monitoring.py                  # SlurmResourceDetector (OTel resource attrs only)
    mlflow_callback.py             # MLflowTrainingCallback (per-epoch metrics + fit-end finalize + LoggedModel)
  _mlflow.py                       # start/end_training_run, log_epoch_metrics, log_test_run, build_search_filter, _dataset_for, _register_logged_model, _upstream_tags
```

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
matching flat flag in `graphids/cli/submit.py` and the `_build_tlas`
helper in `graphids/slurm/submit.py`.

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
auto-loads `./.env` from the project root, so login-node invocations
don't need `set -a; source ./.env`. `extra="ignore"` so shell-only
`GRAPHIDS_*` vars in `.env` (read by `_preamble.sh` etc.) don't trip
validation.

Two distinct path roots — **don't conflate**:

- **`GRAPHIDS_LAKE_ROOT`** — shared lake (cross-user) for `mlflow.db`,
  `cache/`, `mlartifacts/`, `slurm_logs/`. Read by `_mlflow.py`,
  `_dataset_for`, `_preamble.sh`. On OSC: `/fs/ess/PAS1266/graphids`.
- **`GRAPHIDS_RUN_ROOT`** — per-user root for run_dirs / checkpoints /
  traces / predictions. Read by `paths.run_dir / vgae_ckpt /
  states_dir`. On OSC: `${LAKE_ROOT}/dev/${USER}`. Both required;
  conflating them is what produced the 2026-04-24 drift between Python
  settings and the (now-deleted) jsonnet preset TLA defaults.

Other envs: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`,
`SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`, `GRAPHIDS_FORCE_RESUME`,
`GRAPHIDS_ALLOW_FALLBACK_BUDGET`, `GRAPHIDS_BUDGET_SAFETY_MARGIN`,
`GRAPHIDS_VRAM_DRIFT_THRESHOLD`.

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

Two stores: **MLflow** owns run-level metadata + per-epoch scalar metrics
timeseries + device telemetry; **OTel** owns `traces.jsonl` for the
`training.fit` span and structured-log events.

- **MLflow run lifecycle** (`graphids/_mlflow.py`): `start_training_run`
  opens the run at fit-start (SQLite backend at `{LAKE_ROOT}/mlflow.db`,
  artifacts under `{LAKE_ROOT}/mlartifacts/`), in per-axis experiment
  `graphids/{dataset}/{group}`. **MLflow is a hard dep — failures
  propagate** (since 2026-04-24). The only soft-failure paths are
  `MlflowException` on resume `log_params` conflict (immutable-param
  rule when config drifted) and `end_training_run` cleanup (logged-not-
  raised so secondary failure can't shadow primary training exception
  via `__context__`). **Idempotent**: status-gated resume on matching
  `run_name` + `phase=fit` (FAILED/KILLED resume; TERMINATED → new;
  RUNNING/FINISHED refuse unless `GRAPHIDS_FORCE_RESUME=1`; git-SHA
  change → new). Logs params, identity tags via `_build_tags`
  (identity + SLURM + git SHA + `uv.lock` hash + python version +
  upstream-teacher `run_dir`/`ckpt_path` tags for presets with upstream
  checkpoints), `mlflow.log_input(MetaDataset(...))` for dataset lineage,
  and the system-metrics sampler (psutil + nvidia-ml-py, 5s interval).
- **`MLflowTrainingCallback`** (`core/mlflow_callback.py`): appends
  `train_loss`/`val_loss`/`lr`/`early_stop.wait` at `step=epoch`; at
  `on_fit_end` stamps `peak_vram_mb` + `epochs_run` + ckpt SHA256,
  registers a metadata-only `LoggedModel` via
  `MlflowClient.create_logged_model(source_run_id=..., model_type='{group}_{variant}', tags={ckpt_path, run_dir, sha256})`,
  and closes the run FINISHED / FAILED.
- **`log_test_run`** (`_mlflow.py`): self-contained test-phase sink,
  **always-fresh** (new `run_id` each call, no resume). Shares `run_name`
  with the fit row; distinguished by `graphids.phase` tag. `compare.py`
  dedups to latest FINISHED per `(variant, seed)`.
- **OTel `traces.jsonl`** (`{run_dir}/`): single `training.fit` span +
  structured-log events (`budget_probed`, `vram_drift_detected`, etc).
  Parsed by `run_io.load_traces`.
- Query MLflow via `mlflow.search_runs(filter_string=build_search_filter(...))` or
  `client.get_metric_history(run_id, key)`. Never hand-compose filter
  strings — `graphids._mlflow.build_search_filter` is the one entry point
  so all graphids identity fields (dataset, group, variant, seed, phase,
  cluster, run_name, run_dir, status) stay consistent across callers.
