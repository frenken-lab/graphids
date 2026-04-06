# GraphIDS Config System

Jsonnet composition + Pydantic validation + direct Lightning instantiation.
Phase 1 (2026-04-05) replaced the 3-chain YAML + `merge_yaml_chain` plumbing
with a single `render_config(jsonnet_path, tla) → dict` call. Phase 2
(2026-04-05) added `validate_config` as the typed Pydantic gate. Phase 3
(2026-04-05) deleted `LightningCLI` / `GraphIDSCLI` / `schema_parser` /
`build_cli` and replaced them with `graphids.core.instantiate.instantiate`
— importlib-based class_path instantiation with signature-filtered
link_arguments. Phase 4 retools the analyzer CLI to keep jsonargparse
with Jsonnet-backed configs (`commands/analyze.py`) per
`docs/migration_plan.md`.

## Architecture

1. Recipe YAML + pipeline topology declare which stage+scale chains exist.
2. `enumerate_assets` produces `StageConfig`s carrying `jsonnet_path` +
   planner-derived identity knobs.
3. `ConfigResolver.resolve` packs trainer/stage/KD/upstream-ckpt overrides
   into a typed TLA dict via `graphids.orchestrate.contracts.build_tla_dict`.
4. `render_config(spec.jsonnet_path, spec.jsonnet_tla)` produces the merged
   dict — identical on login node, dagster worker, and SLURM node.
5. `validate_config(rendered) → ValidatedConfig` runs Pydantic validators
   (null list fields, monitor consistency, class_path namespacing, etc.)
   on the jsonnet output. Raises `ConfigValidationError` on any violation.
6. `graphids.core.instantiate.instantiate(rendered, validated=...) →
   InstantiatedRun` imports class_paths via `importlib`, applies
   signature-filtered link_arguments, builds forced callbacks, and returns
   a wired `(trainer, model, datamodule)` triple.
7. Full tree: `docs/reference/config-architecture.md`.

## File layout

```
configs/                           # repo root — jsonnet sources
├── _lib/
│   ├── defaults.libsonnet         # trainer / checkpoint / early_stopping defaults
│   └── helpers.libsonnet          # apply_dotted() for recipe overrides
├── datasets/
│   └── dataset_registry.json      # dataset catalog (domain → dataset metadata)
├── matrix/
│   ├── axes.json                  # valid model types / scales / fusion methods
│   └── topology.json              # stage DAG + identity keys
├── recipes/                       # pipeline recipes (sweep dimensions)
├── stages/
│   ├── autoencoder.jsonnet        # function(dataset, seed, run_dir, scale, conv_type, ...)
│   ├── normal.jsonnet
│   ├── curriculum.jsonnet
│   ├── fusion.jsonnet
│   └── analyze_{vgae,gat,fusion}.jsonnet  # Analyzer configs (NOT in CLI chain)
├── models/
│   ├── vgae.libsonnet             # { base, scales: {small, large}, kd }
│   ├── gat.libsonnet
│   └── dgi.libsonnet
├── resources/
│   ├── clusters.json              # cluster → partition/gres mapping
│   ├── job_profiles.json          # per-family/scale/stage resource sizing
│   └── submit_profiles.json       # scripts/slurm/submit.sh profiles
├── fusion.libsonnet               # { base, methods: {bandit, dqn, mlp, weighted_avg} }
└── fusion/
    ├── base.libsonnet
    └── methods/{bandit,dqn,mlp,weighted_avg}.libsonnet

graphids/
  callbacks.py                     # ResourceProfileCallback, RunRecordCallback (pl.Callback)
  commands/
    train.py                       # fit/test/validate/predict — argparse + instantiate
    # plus operational subcommands (analyze, profile, from-spec, ...)
  config/
    __init__.py                    # public API facade
    base.py                        # CONFIG_DIR, PROJECT_ROOT
    runtime.py                     # env vars, constants
    topology.py                    # stage DAG + import-time jsonnet-tree assertions
    paths.py                       # run_dir, compute_identity_hash
    contracts.py                   # TrainingRunConfig, KDEntry, expand_recipe_configs
    jsonnet.py                     # render_config(path, tla) subprocess shim
    yaml_utils.py                  # read_yaml / write_yaml (snapshots only)
  core/
    contracts/
      models.py                    # TrainingSpec — jsonnet_path, jsonnet_tla, identity
      ops.py                       # build_tla_dict, resolve_jsonnet_path (training spec helpers)
      run_record.py                # RunRecord sidecar schema
    instantiate.py                 # instantiate(rendered) → InstantiatedRun (trainer, model, datamodule)
    train_entrypoint.py            # render_config → validate_config → snapshot → instantiate → fit
  orchestrate/
    planning.py                    # enumerate_assets (StageConfig in config/shared.py)
    resolve.py                     # ConfigResolver, cross-field validation via config/schemas.py
    component.py                   # SlurmTrainingComponent (dagster)
    assets.py                      # @asset definitions
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

# Pipeline path (dagster → SLURM)
dg launch --assets 'autoencoder_*'

# Validation runs inside ConfigResolver during orchestration
```

## Stage function convention

Every `stages/*.jsonnet` is a top-level function with sensible defaults
for every TLA. `graphids.orchestrate.contracts.build_tla_dict` is the single site that
packs a `StageConfig` into the TLA dict each stage consumes. Adding a new
TLA means updating both the jsonnet signature AND `build_tla_dict`.

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
6. **`ConfigResolver.resolve`** — renders every unique chain on asset
   materialization, runs `validate_config` + `validate_stage_config` (Pydantic
   cross-field rules), and catches override typos, null list fields, and
   logger/callback wiring mismatches. No jsonargparse schema pass — deleted
   in Phase 3.

## Null preservation

`data.init_args.num_workers: null` is a real value (auto-sized from
GPU-first sizing), not "missing". Jsonnet has a first-class `null` —
preserve it. The autoencoder/curriculum stages emit `num_workers: null`
explicitly; `gat.base.libsonnet` overrides it to `4` because GAT is
compute-bound.

## Environment variables

Infrastructure env vars use `os.environ.get()` in `runtime.py` with
`KD_GAT_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.
The `identity_hash` suffix is an 8-char SHA256 derived from the stage's
`identity_keys` (defined in `topology.py`). Computed by
`compute_identity_hash()` in `paths.py`. **Missing identity keys raise
`KeyError`** — never silently hash to defaults.

## Run record sidecars

Every training run writes `{run_dir}/run_record.json` — a structured JSON
sidecar that is the source of truth for experiment status and metrics.
Written atomically (temp + fsync + rename).

- **`RunRecord`** Pydantic model in `core/contracts/run_record.py`
- **`RunRecordCallback`** in `graphids/callbacks.py` — writes sidecar on
  `on_fit_start`/`on_fit_end`/`on_exception`
- **`_finalize-record`** command — called in generated sbatch script
  after test+analyze to add phase markers + sacct wall_time

## DuckDB catalog

`{lake_root}/catalog/kd_gat.duckdb` — `runs` table rebuilt from
`run_record.json` sidecars. Disposable — rebuildable via
`python -m graphids rebuild-catalog`.
