# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct instantiation.
Replaced YAML + LightningCLI with
`render_config(jsonnet_path, tla) → dict` → `validate_config` (Pydantic) →
`graphids.instantiate.instantiate` (importlib class_path instantiation with
signature-filtered link_arguments). PyTorch Lightning was removed in favor
of a custom `graphids.core.trainer.Trainer`. Analyzer CLI uses jsonargparse
with Jsonnet-backed configs (`cli/_analysis.py`).

## Architecture

1. Recipe YAML + pipeline topology declare which stage+scale chains exist.
2. `enumerate_assets` produces `StageConfig`s carrying `jsonnet_path` +
   planner-derived identity knobs.
3. `ResolvedConfig.resolve` (classmethod on `orchestrate/resolve.py`)
   packs trainer/stage/KD/upstream-ckpt overrides into a typed TLA
   dict via the private `_build_tla_dict` helper in the same file.
4. `render(stage.jsonnet_path, tla)` from `graphids.config.jsonnet`
   produces the merged dict — identical on login node and SLURM node.
5. `validate_config(rendered) → ValidatedConfig` runs Pydantic validators
   (null list fields, monitor consistency, class_path namespacing, etc.)
   on the jsonnet output. Raises `ConfigValidationError` on any violation.
6. `graphids.instantiate.instantiate(rendered, validated=...) →
   InstantiatedRun` imports class_paths via `importlib`, applies
   signature-filtered link_arguments, builds forced callbacks, and returns
   a wired `(trainer, model, datamodule)` triple.
7. Full tree: `docs/reference/config-architecture.md`.

## File layout

```
configs/                           # repo root — jsonnet sources
├── _lib/
│   ├── defaults.libsonnet         # trainer / checkpoint / early_stopping defaults
│   ├── helpers.libsonnet          # apply_dotted() for recipe overrides
│   └── recipes.libsonnet          # recipe expansion logic
├── datasets/
│   └── dataset_registry.json      # dataset catalog (domain → dataset metadata)
├── matrix/
│   ├── axes.json                  # valid model types / scales / fusion methods
│   └── topology.json              # stage DAG + identity keys
├── recipes/                       # pipeline recipes (sweep dimensions)
├── stages/
│   ├── autoencoder.jsonnet        # function(dataset, seed, run_dir, scale, conv_type, ...)
│   ├── supervised.jsonnet         # (was normal.jsonnet + curriculum.jsonnet)
│   ├── fusion.jsonnet
│   └── analyze_{vgae,gat,fusion}.jsonnet  # Analyzer configs (NOT in CLI chain)
├── models/
│   ├── unsupervised.libsonnet     # { base, scales: {small, large}, kd } (was vgae + dgi)
│   └── supervised.libsonnet       # (was gat)
├── resources/
│   ├── clusters.json              # cluster → partition/gres mapping
│   ├── job_profiles.json          # per-family/scale/stage resource sizing
│   └── submit_profiles.json       # scripts/slurm/submit.sh profiles
├── fusion.libsonnet               # { base, methods: {bandit, dqn, mlp, weighted_avg} }
└── fusion/
    ├── base.libsonnet
    └── methods/{bandit,dqn,mlp,weighted_avg}.libsonnet

graphids/
  instantiate.py                   # instantiate(rendered) → InstantiatedRun (trainer, model, datamodule)
  cli/
    app.py                         # Typer root app, shared options (parse_tla, apply_overrides)
    _training.py                   # fit/test/validate/predict commands
    _analysis.py                   # analyze command (jsonargparse + jsonnet)
    _data.py                       # rebuild-caches, stage-data, extract-fusion-states
    _monarch.py                    # monarch-run (pipeline execution)
    _slurm.py                      # job-stats, submit-profile
  config/
    __init__.py                    # public API facade
    constants.py                   # CONFIG_DIR, PROJECT_ROOT, env var defaults
    topology.py                    # stage DAG + import-time jsonnet-tree assertions
    paths.py                       # run_dir, compute_identity_hash
    schemas.py                     # validate_config → ValidatedConfig (Pydantic)
    jsonnet.py                     # render_config(path, tla) subprocess shim
  core/
    trainer.py                     # Trainer, TrainerConfig, seed_everything, MetricAccumulator
    callbacks.py                   # CallbackBase, ModelCheckpoint, EarlyStopping
    monitoring.py                  # OTelTrainingCallback, OTelTrainingLogger
  orchestrate/
    run.py                         # PipelineConfig, build_pipeline_stages, run_pipeline driver
    allocate.py                    # JobSpec, build_slurm_job, spawn_actor, configure_monarch
    chain.py                       # run_chain(actor, stages, …) — pure loop + ChainResult
    stage.py                       # build, train, evaluate, run_stage (single-stage primitives)
    analyze.py                     # pipeline-level analyze + run_single_analysis helper
    actors.py                      # PipelineActor — thin endpoint wrapper around stage.py
    resolve.py                     # ResolvedConfig.resolve + private _build_tla_dict
    _setup.py                      # ensure_spawn, touch_marker
    planning/
      planner.py                   # StageConfig, enumerate_assets, resolve_jsonnet_path
      recipes.py                   # TrainingRunConfig, KDEntry, expand_recipe_configs
  slurm/
    env.py                         # centralized SLURM env var reads
    core/
      accounting.py                # sacct wrappers
      submit.py                    # sbatch submission
    ops/
      profile.py                   # resource profiling
      staging.py                   # NFS → scratch → TMPDIR staging
    pipeline.py                    # GraphIDS-specific spec → SLURM plumbing
```

## Running

```bash
# Dev path (no TLAs needed — stages default for smoke)
python -m graphids fit --config configs/stages/autoencoder.jsonnet

# Dev path with TLAs (preprocessor harvests them and passes to render_config)
python -m graphids fit \
    --tla 'dataset="hcrl_sa"' \
    --tla 'scale="large"' \
    --tla 'variational=false' \
    --config configs/stages/autoencoder.jsonnet \
    --model.init_args.lr=0.005

# Pipeline path (Monarch → single SLURM allocation)
python -m graphids monarch-run --dataset hcrl_sa

# Validation runs inside ResolvedConfig.resolve during orchestration
```

## Stage function convention

Every `stages/*.jsonnet` is a top-level function with sensible defaults
for every TLA. The private `_build_tla_dict` helper in
`graphids/orchestrate/resolve.py` is the single site that packs a
`StageConfig` into the TLA dict each stage consumes. Adding a new TLA
means updating both the jsonnet signature AND `_build_tla_dict`.

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
`~/.local/bin/jsonnet configs/stages/<stage>.jsonnet` to verify a stage
renders correctly after editing.

## Robustness

1. **Typed TLA round-trip.** `render_config` passes every TLA through
   `--tla-code <k>=<json.dumps(v)>` so ints stay ints, bools stay bools,
   lists stay lists, `None` becomes jsonnet `null`.
2. **Pydantic gate (`ValidatedConfig`)** — runs immediately after
   `render_config`: null list fields in `model.init_args`, monitor
   mismatch between `checkpoint` and `early_stopping`, un-namespaced
   `class_path` strings, and `LearningRateMonitor` without `logger`
   all die with an actionable error before any torch import.
3. **KD auxiliaries → `SimpleNamespace`** — `instantiate.py` coerces
   `model.init_args.auxiliaries` list items from `dict` to
   `SimpleNamespace` so `_install_kd_teacher`'s attribute-access
   contract (`getattr(a, "type", None)`) keeps working without the
   jsonargparse TypedDict+Namespace dance.
4. **Signature-filtered link_arguments** — `_apply_link_arguments`
   inspects the target class's `__init__` signature and silently skips
   links whose target name isn't in the accepted kwarg set. Fusion
   models (`BanditFusionModule` / `DQNFusionModule` / `MLPFusionModule` /
   `WeightedAvgModule`) don't take `dataset` / `conv_type` / `heads` /
   `seed` / `lake_root` so every VGAE/GAT link is a no-op for them.
5. **`topology.py` import-time assertions** — cross-validates the jsonnet
   tree: every `(model_family)` has a libsonnet, every stage has a
   `.jsonnet`, every fusion method has a method libsonnet. Missing files
   fail at package import.
6. **`ResolvedConfig.resolve`** — for each stage on actor invocation,
   renders the jsonnet, runs `validate_config` (Pydantic — null list
   fields, monitor consistency, class_path namespacing, logger/callback
   wiring), and emits an inline log warning if the checkpoint
   `monitor`/`mode` pair doesn't match the stage family convention.
   No jsonargparse schema pass — deleted in Phase 3.

## Null preservation

`data.init_args.num_workers: null` is a real value (auto-sized from
GPU-first sizing), not "missing". Jsonnet has a first-class `null` —
preserve it. The autoencoder stage emits `num_workers: null`
explicitly; `supervised.libsonnet` overrides it to `4` because GAT is
compute-bound.

## Environment variables

Infrastructure env vars use `os.environ.get()` in `config/constants.py`
and `slurm/env.py` with `KD_GAT_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.
The `identity_hash` suffix is an 8-char SHA256 derived from the stage's
`identity_keys` (defined in `topology.py`). Computed by
`compute_identity_hash()` in `paths.py`. **Missing identity keys raise
`KeyError`** — never silently hash to defaults.

## Observability (OpenTelemetry)

Every training run writes `{run_dir}/traces.jsonl` + `{run_dir}/metrics.jsonl`
via OTel SimpleSpanProcessor + PeriodicExportingMetricReader. The
`training.fit` span is the source of truth for experiment status + metrics.

- **`OTelTrainingCallback`** in `core/monitoring.py` — creates span on
  `on_fit_start`, closes on `on_fit_end`/`on_exception`, records per-batch
  VRAM + timing as histograms/gauges
- **`OTelTrainingLogger`** in `core/monitoring.py` — emits `model.log()`
  metrics as OTel histograms

## DuckDB catalog

`{lake_root}/catalog/kd_gat.duckdb` — intended to hold a `runs` table
built from `traces.jsonl` OTel spans. Builder + `rebuild-catalog` CLI
were removed 2026-04-10 pending redesign — `traces.jsonl` files are
still written and can be queried ad-hoc via DuckDB's `read_json_auto`.
