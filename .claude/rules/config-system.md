# GraphIDS Config System

LightningCLI + jsonargparse + plain YAML. No Hydra, no OmegaConf, no dataclasses.

## Architecture

jsonargparse reads `__init__` signatures on LightningModules and DataModules for type info. YAML provides values. CLI overrides win.

```
defaults/trainer.yaml (shared defaults) → stage YAML (model class_path + overrides) → model scale YAML (dims) → CLI
```

## File layout

```
graphids/
  cli.py               # GraphIDSCLI + CLI_KWARGS — shared by __main__ and orchestrate
  commands/             # operational subcommands (registered in __main__.py _COMMAND_MODULES)
    analyze.py           # analysis artifacts from checkpoints
    analyze_from_spec.py # run analyzer from canonical AnalysisSpec (dagster transport)
    landscape.py         # 2D loss landscape
    profile.py           # sacct resource profiler
    profile_training.py  # profiled training run (PyTorchProfiler)
    rebuild_caches.py    # rebuild preprocessed graph caches
    stage_data.py        # NFS → scratch → TMPDIR staging
    submit_profile.py    # print SLURM resource profile for submit.sh
    test_preprocessing.py # validate preprocessing pipeline
    train_from_spec.py   # run training from canonical TrainingSpec (dagster transport)
    _spec_payload.py     # shared spec deserialization for *_from_spec commands
  config/
    __init__.py          # re-export facade (public API: constants, topology, paths, contracts)
    base.py              # CONFIG_DIR, PROJECT_ROOT
    runtime.py           # env vars, constants (from old constants.yaml)
    topology.py          # stage DAG, valid types/scales (from old pipeline.yaml)
    paths.py             # run_dir(), load_catalog(), dataset_names(), compute_identity_hash()
    contracts.py         # TrainingRunConfig, KDEntry, expand_recipe_configs()
    yaml_utils.py        # read_yaml()
    recipe_expand.py     # recipe expansion logic
    defaults/            # shared trainer + global defaults
      trainer.yaml       # seed, trainer (callbacks, precision, etc.)
      global.yaml        # global defaults (from old constants.yaml static values)
      io.yaml            # I/O defaults (from old constants.yaml + write_paths.yaml)
    datasets/            # per-dataset catalog (from old monolithic datasets.yaml)
      hcrl_ch.yaml
      hcrl_sa.yaml
      set_01.yaml ... set_04.yaml
    matrix/              # pipeline axes and constraints
      axes.yaml          # valid model types, scales, fusion methods
    resources/           # SLURM resource profiles (from old monolithic resources.yaml)
      clusters.yaml      # cluster-specific settings (partitions, GPUs)
      submit_profiles.yaml # submit.sh profile mappings
      profiles/          # per-model resource profiles
        vgae.yaml, gat.yaml, dgi.yaml, temporal.yaml, fusion.yaml
    stages/              # one per stage — model class_path + init_args overrides + data
      autoencoder.yaml   # VGAEModule + CANBusDataModule
      normal.yaml        # GATModule + CANBusDataModule (no curriculum)
      curriculum.yaml    # GATModule + CurriculumDataModule
      temporal.yaml      # TemporalLightningModule + TemporalDataModule
      fusion.yaml        # single fusion stage YAML (all methods)
      analyze_vgae.yaml  # Analyzer config: VGAE embeddings + landscape
      analyze_gat.yaml   # Analyzer config: GAT embeddings + attention + CKA + landscape
      analyze_fusion.yaml # Analyzer config: fusion policy
    fusion/              # fusion-specific config (method overlays + scales)
      base.yaml          # shared fusion defaults
      methods/           # per-method overlays (bandit, dqn, mlp, weighted_avg)
      scales/            # fusion scale configs (small, large)
    models/              # per-model architecture configs (base + scales)
      vgae/
        base.yaml        # shared VGAE architecture defaults
        scales/
          small.yaml, large.yaml
      gat/
        base.yaml        # shared GAT architecture defaults
        scales/
          small.yaml, large.yaml
      dgi/
        base.yaml
        scales/
          small.yaml, large.yaml
      temporal/
        base.yaml
        scales/
          small.yaml, large.yaml
    recipes/             # run specifications (sweep dimensions, config overrides)
      ablation.yaml, final_eval.yaml, smoke_test.yaml
  core/
    models/
      __init__.py        # re-exports
      _conv.py           # shared conv building blocks
      _training.py       # KDAuxiliary TypedDict, shared training utils
      autoencoder/       # VGAE, DGI
        vgae.py, dgi.py
      supervised/        # GAT
        gat.py
      temporal_family/   # temporal transformer
        temporal.py
      fusion/            # all fusion models
        bandit.py, dqn.py, fusion_baselines.py, fusion_features.py, fusion_reward.py
    contracts/           # canonical specs for dagster↔SLURM transport
      analysis.py, models.py, ops.py
  orchestrate/
    component.py         # integration hub — SlurmTrainingComponent + IOManager + Resource
    definitions.py       # dagster entry point
    __main__.py          # CLI: run/validate/smoke
    planning.py          # recipe → execution plan
    execution.py         # plan executor
    assets.py            # dagster asset definitions
    checks.py            # dagster freshness/quality checks
    analysis.py          # analysis asset integration
    validate.py          # config chain validation
    slurm.py             # sbatch submit, sacct poll
    resources.py         # ResourceSpec + scale_resources
```

`LAKE_ROOT` defaults to `experimentruns` (relative) or `KD_GAT_LAKE_ROOT` env var (ESS on OSC).

## CLI usage

`GraphIDSCLI` class and `CLI_KWARGS` in `graphids/cli.py`. Commands registered via `_COMMAND_MODULES` dict in `__main__.py`:

```bash
# --- Training (GraphIDSCLI → LightningCLI) ---
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids fit --config graphids/config/stages/autoencoder.yaml \
                       --config graphids/config/models/vgae/scales/small.yaml
python -m graphids fit --config graphids/config/stages/normal.yaml \
                       --model.init_args.lr=0.01

# --- Fusion (single stage YAML + method overlay) ---
python -m graphids fit --config graphids/config/stages/fusion.yaml \
                       --config graphids/config/fusion/methods/bandit.yaml

# --- Analysis artifacts (Analyzer — no Trainer) ---
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa

# --- Spec-file transport (dagster → SLURM) ---
python -m graphids train-from-spec --spec-file /tmp/spec.json
python -m graphids analyze-from-spec --spec-file /tmp/spec.json
```

`analyze` YAML keys nest under `analyzer:` (same pattern as `model:`/`data:`/`trainer:`). Required args (`ckpt_path`, `dataset`, `model_type`) have no defaults — jsonargparse rejects configs that omit them.

`train-from-spec` and `analyze-from-spec` accept a canonical spec file (JSON) produced by dagster's `SlurmTrainingComponent`. This is the transport layer for dagster→SLURM job submission — dagster serializes the spec, SLURM deserializes and runs it.

## Model __init__ convention

Every LightningModule takes **flat typed primitives** — no nested config objects. jsonargparse introspects the signature; YAML maps directly to init_args.

```python
class VGAEModule(pl.LightningModule):
    def __init__(self, conv_type: str = "gatv2", hidden_dims: list[int] | None = None,
                 latent_dim: int = 48, lr: float = 0.003,
                 auxiliaries: list[KDAuxiliary] | None = None, ...):
        self.save_hyperparameters()
```

**Structured list items** use `TypedDict` for jsonargparse validation. `KDAuxiliary` (in `_training.py`) defines valid keys for KD config — typos in YAML are rejected at parse time.

```yaml
# stages/autoencoder.yaml — keys match __init__ params exactly
model:
  class_path: graphids.core.models.autoencoder.vgae.VGAEModule
  init_args:
    proj_dim: 48
    lr: 0.002
```

**Prefix conventions** for modules with colliding param spaces:
- `TemporalLightningModule`: `spatial_*` for GAT backbone, `temporal_*` for transformer
- `DQNFusionModule` / `BanditFusionModule`: separate classes, no prefix needed (each has its own params)

## Pipeline topology

`topology.py` defines model types, scales, fusion methods, stages, DAG dependencies, and variants in Python (migrated from old `pipeline.yaml`). `__init__.py` re-exports `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`, `VALID_FUSION_METHODS`. `matrix/axes.yaml` declares the valid combinations for recipe expansion.

Import-time assertions in `topology.py` cross-validate the topology against `config/models/` and `config/resources/profiles/`:
every `(model_type, scale)` and `(fusion_method, scale)` must have model config files;
every trainable `(model, scale, stage)` must have a resource profile.

## Environment variables

Infrastructure env vars use `os.environ.get()` in `runtime.py` with `KD_GAT_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

jsonargparse also supports `--env_prefix=KD_GAT` for any init_args field.

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.

The `identity_hash` suffix is an 8-char SHA256 derived from the stage's `identity_keys` (defined in `topology.py`). Computed by `compute_identity_hash()` in `paths.py`. **Missing identity keys raise `KeyError`** — never silently hash to defaults.

## Config robustness

Four layers of validation prevent silent config drift:

1. **jsonargparse type checking** — unknown `init_args` keys and wrong types are rejected at parse time.
2. **`KDAuxiliary` TypedDict** — structured list items (KD config) validate keys at parse time. Typos like `alppha` are caught.
3. **`topology.py` import-time assertions** — cross-validates model types, scales, and resource profiles at import time. Missing config files or resource profiles raise immediately.
4. **`python -m graphids.orchestrate validate`** — checks config chains parse, no callback/logger incompatibility, no null list fields in model init_args.

When adding new config fields: type annotations on `__init__` params are the schema. Use `TypedDict` for structured dicts/lists. jsonargparse enforces the rest.

**Null-serialization rule:** `--print_config` serializes `Optional[X] = None` defaults as `null`. If `__init__` normalizes `None → real_default` before `save_hyperparameters()`, the stage YAML MUST set the field explicitly so expanded configs never contain `null`. Grep pattern to audit: `if .* is None:` before `save_hyperparameters()` in any LightningModule.

## DuckDB catalog

`{lake_root}/catalog/kd_gat.duckdb` — `experiments` table with flat metric columns + `config JSON` + `identity_hash`. Best-effort, disposable — rebuildable from filesystem.
