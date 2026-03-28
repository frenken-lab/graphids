# KD-GAT Config System

LightningCLI + jsonargparse + plain YAML. No Hydra, no OmegaConf, no dataclasses.

## Architecture

jsonargparse reads `__init__` signatures on LightningModules and DataModules for type info. YAML provides values. CLI overrides win.

```
trainer.yaml (shared defaults) → stage YAML (model class_path + overrides) → overlay YAML (scale) → CLI
```

## File layout

```
graphids/config/
  __init__.py          # constants, topology, path helpers (single Python file)
  constants.yaml       # static values: preprocessing_version, SLURM defaults, ckpt mappings
  pipeline.yaml        # DAG topology: stages, dependencies, identity_keys, valid models/scales
  datasets.yaml        # dataset catalog (YAML anchors for shared configs)
  resources.yaml       # SLURM resource profiles per model×scale×stage
  trainer.yaml         # default_config_files: seed, trainer (callbacks, precision, etc.)
  stages/              # one per stage — model class_path + init_args overrides + data
    autoencoder.yaml   # VGAEModule + CANBusDataModule
    normal.yaml        # GATModule + CANBusDataModule (no curriculum)
    curriculum.yaml    # GATModule + CurriculumDataModule
    fusion.yaml        # RLFusionModule + FusionDataModule + trainer overrides
    analyze_vgae.yaml  # Analyzer config: VGAE embeddings + landscape
    analyze_gat.yaml   # Analyzer config: GAT embeddings + attention + CKA + landscape
    analyze_fusion.yaml # Analyzer config: fusion policy
  overlays/            # thin --config adds for scale/ablation variants
    small_vgae.yaml    # small-scale VGAE dims
    small_gat.yaml     # small-scale GAT dims
    small_dgi.yaml     # small-scale DGI dims
```

## CLI usage

Two entry points in `__main__.py`, both using jsonargparse:

```bash
# --- Training (GraphIDSCLI → LightningCLI) ---
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids fit --config graphids/config/stages/autoencoder.yaml \
                       --config graphids/config/overlays/small_vgae.yaml
python -m graphids fit --config graphids/config/stages/normal.yaml \
                       --model.init_args.lr=0.01

# --- Analysis artifacts (Analyzer — no Trainer) ---
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

`analyze` YAML keys nest under `analyzer:` (same pattern as `model:`/`data:`/`trainer:`). Required args (`ckpt_path`, `dataset`, `model_type`) have no defaults — jsonargparse rejects configs that omit them.

## Model __init__ convention

Every LightningModule takes **flat typed primitives** — no nested config objects. jsonargparse introspects the signature; YAML maps directly to init_args.

```python
class VGAEModule(pl.LightningModule):
    def __init__(self, conv_type: str = "gatv2", hidden_dims: list[int] | None = None,
                 latent_dim: int = 48, lr: float = 0.003, ...):
        self.save_hyperparameters()
```

```yaml
# stages/autoencoder.yaml — keys match __init__ params exactly
model:
  class_path: graphids.core.models.vgae.VGAEModule
  init_args:
    proj_dim: 48
    lr: 0.002
```

**Prefix conventions** for modules with colliding param spaces:
- `TemporalLightningModule`: `spatial_*` for GAT backbone, `temporal_*` for transformer
- `RLFusionModule`: `dqn_*` for DQN agent, `bandit_*` for bandit agent

## Pipeline topology

`pipeline.yaml` defines model types, scales, stages, DAG dependencies, and variants. `__init__.py` loads this once and exposes `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`.

## Environment variables

Infrastructure env vars use `os.environ.get()` in `__init__.py` with `KD_GAT_` prefix:

- SLURM: `SLURM_ACCOUNT`, `SLURM_PARTITION`, `SLURM_GPU_TYPE`
- Run metadata: `SWEEP_ID`, `USER_TAGS`, `CKPT_PATH`

jsonargparse also supports `--env_prefix=KD_GAT` for any init_args field.

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.

The `identity_hash` suffix is an 8-char SHA256 derived from the stage's `identity_keys` (defined in `pipeline.yaml`). Computed by `compute_identity_hash()` in `__init__.py`.

## DuckDB catalog

`{lake_root}/catalog/kd_gat.duckdb` — `experiments` table with flat metric columns + `config JSON` + `identity_hash`. Written by `_append_to_catalog()` in `graphids/pipeline/stages/__init__.py` after each stage completes. Best-effort. Catalog is disposable — rebuildable from filesystem.
