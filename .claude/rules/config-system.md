# KD-GAT Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**.

## File layout (5 YAML + 2 Python)

```
graphids/config/
  __init__.py       # resolve() + re-exports + path helpers (from old paths.py)
  constants.py      # Python constants + topology from pipeline.yaml
  config.yaml       # Hydra root: ALL defaults inline (no config groups)
  models.yaml       # Model×scale presets, keyed by {model_type}_{scale}
  pipeline.yaml     # DAG topology (stages, variants, dependencies)
  datasets.yaml     # Dataset catalog
  resources.yaml    # SLURM resource profiles
```

No config groups, no subdirectories. Model presets in `models.yaml` are merged by Python after Hydra compose.

## Composition order

```
config.yaml (all defaults) → model preset from models.yaml → CLI overrides
```

`resolve()` and `_merge_model_preset()` handle this merge. CLI overrides always win.

## CLI usage

```bash
python -m graphids stage=autoencoder model_type=vgae scale=large dataset=hcrl_sa
python -m graphids stage=autoencoder model_type=vgae scale=large training.lr=0.001
python -m graphids --multirun stage=autoencoder model_type=vgae scale=large training.lr=0.001,0.01
```

## Pipeline topology

`pipeline.yaml` defines model types, scales, stages, DAG dependencies, and variants. `constants.py` loads this once and exposes `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`. Variants are read directly from `pipeline.yaml` by `dag.py`.

## Environment variables

Path vars (`lake_root`) flow through Hydra `oc.env` resolvers in `config.yaml`. Infrastructure env vars are plain `os.environ.get()` calls in `__init__.py` with `KD_GAT_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.

The `identity_hash` suffix is an 8-char SHA256 derived from the stage's `identity_keys` (defined in `pipeline.yaml`). It prevents run directory collisions between ablation configs that share the same model_type+scale+stage. Computed by the `identity_hash` custom OmegaConf resolver registered in `graphids/config/__init__.py`.

## DuckDB catalog

`{lake_root}/catalog/kd_gat.duckdb` — `experiments` table with flat metric columns + `config JSON` + `identity_hash`. Written by `_append_to_catalog()` in `graphids/pipeline/stages/__init__.py` after each stage completes. Best-effort (never fails the training job). Catalog is disposable — rebuildable from filesystem.
