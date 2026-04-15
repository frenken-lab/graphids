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

One route. `scripts/run <preset.jsonnet>` or `python -m graphids fit` →

1. `render(config_path, tla)` from `graphids.config.jsonnet` renders the
   merged dict. Every preset under `configs/ablations/` is a top-level
   function that computes its own `run_dir` from `(lake_root, dataset,
   seed)` via `configs/ablations/_paths.libsonnet`.
2. `apply_overrides(rendered, --set ...)` applies dotted-path overrides
   in-place on the rendered dict (`cli/app.py`).
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
`SBATCH_DEP=afterok:<jid>` — see `scripts/ablation/launch_set_01.sh`.
No in-process pipeline driver, no planner, no identity-hash layer.

Full tree: `docs/reference/config-architecture.md`.

## File layout

```
configs/
├── _lib/
│   ├── defaults.libsonnet         # trainer / checkpoint / early_stopping defaults
│   └── helpers.libsonnet          # apply_dotted() for trainer/stage overrides
├── ablations/
│   ├── _paths.libsonnet           # run_dir / ckpt / states_dir derivations
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
│   └── submit_profiles.json       # static + scaling profiles for submit.sh
├── fusion.libsonnet               # { base, methods: {bandit, dqn, mlp, weighted_avg} }
└── fusion/
    ├── base.libsonnet
    └── methods/{bandit,dqn,mlp,weighted_avg}.libsonnet

graphids/
  cli/
    app.py                         # Typer root app + shared option types + apply_overrides + submit-profile
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
    constants.py                   # CONFIG_DIR, PROJECT_ROOT, env var defaults
    topology.py                    # stage-file existence check, dataset catalog, path helpers
    schemas.py                     # validate_config → ValidatedConfig (Pydantic)
    jsonnet.py                     # render(path, tla) via _jsonnet C bindings
  core/
    trainer.py                     # Trainer, TrainerConfig, seed_everything
    callbacks.py                   # CallbackBase, ModelCheckpoint, EarlyStopping, VRAMDriftCallback
    monitoring.py                  # OTelTrainingCallback, OTelTrainingLogger
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
scripts/run configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
```

## Stage / ablation function convention

Every `stages/*.jsonnet` and `ablations/**/*.jsonnet` is a top-level
function with sensible defaults for every TLA. Adding a new TLA means
updating the jsonnet signature + (if the TLA is launcher-level) the
`_push_*_tla` dispatch in `scripts/run`.

```jsonnet
function(
  dataset='hcrl_ch', seed=42, run_dir='',
  scale='small', conv_type='gatv2', variational=true,
  auxiliaries=[], vgae_ckpt_path=null,
  trainer_overrides={}, stage_overrides={}, ckpt_path=null,
)
  defaults.trainer + defaults.checkpoint + defaults.early_stopping
  + vgae.base + vgae.scales[scale]
  + { seed_everything: seed, trainer+: {...}, data: {...}, model+: {...} }
  + helpers.apply_dotted(trainer_overrides)
  + helpers.apply_dotted(stage_overrides)
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

Infrastructure env vars use `os.environ.get()` in `config/constants.py`
and `slurm/env.py` with `GRAPHIDS_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

## Path layout

Every ablation preset computes its own `run_dir` via
`configs/ablations/_paths.libsonnet`:

```
{lake_root}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`lake_root` defaults to `experimentruns` when `GRAPHIDS_LAKE_ROOT` is
unset. Path logic lives in jsonnet next to the preset; there is no
Python planner / identity-hash layer.

## Observability (OpenTelemetry)

Every training run writes `{run_dir}/traces.jsonl` + `{run_dir}/metrics.jsonl`
as NDJSON via OTel SimpleSpanProcessor + PeriodicExportingMetricReader.
The `training.fit` span is the source of truth for experiment status +
metrics.

- **`OTelTrainingCallback`** (`core/monitoring.py`) — creates span on
  `on_fit_start`, closes on `on_fit_end` / `on_exception`, records
  per-batch VRAM + timing as histograms/gauges.
- **`OTelTrainingLogger`** (`core/monitoring.py`) — emits `model.log()`
  metrics as OTel histograms.
- **`run_io.load_traces` / `load_metrics`** (`core/run_io.py`) — polars
  parser; flattens OTel histogram/gauge/sum data points into a long-format
  DataFrame.
