# Plan: Flatten Model Config — Kill Dataclasses, Wire jsonargparse Direct

> Created: 2026-03-28
> Config reorg: DONE (2026-03-28)

## Context

The codebase is **intentionally broken**. Commit `8a69c10` deleted `schema.py` (all config dataclasses) and `coerce_config`. Five of seven LightningModules now fail on import because they reference deleted types (`VGAEConfig`, `GATConfig`, `DGIConfig`, `TrainingConfig`) and deleted function (`coerce_config`).

**Goal:** Every LightningModule `__init__` takes flat typed primitives. jsonargparse reads the `__init__` signature for types, YAML provides values. No dataclasses. No coerce_config. No type conversion layer. This is how Lightning was designed to work.

**Already correct:** `MLPFusionModule`, `WeightedAvgModule` — flat primitives, untouched.

## Config Layout (DONE)

```
graphids/config/
  __init__.py            # loads YAML, derives topology, path helpers
  constants.yaml         # project constants (preprocessing, SLURM defaults, ckpt mappings)
  pipeline.yaml          # DAG topology, stages, dependencies, identity_keys
  datasets.yaml          # dataset catalog with YAML anchors
  resources.yaml         # SLURM resource profiles per model×scale×stage
  trainer.yaml           # default_config_files: seed + trainer (callbacks, precision, etc.)
  stages/                # one per stage — model class_path + init_args overrides + data
    autoencoder.yaml     # VGAEModule + CANBusDataModule
    normal.yaml          # GATModule + CANBusDataModule
    curriculum.yaml      # GATModule + CurriculumDataModule
    fusion.yaml          # RLFusionModule + FusionDataModule + trainer overrides
  overlays/              # thin --config adds for scale variants
    small_vgae.yaml      # small-scale VGAE architecture dims
    small_gat.yaml       # small-scale GAT architecture dims
    small_dgi.yaml       # small-scale DGI architecture dims
```

**Deleted:** `base.yaml`, `schema.yaml`, `small.yaml`, old top-level stage YAMLs, entire `defaults/` subdirectory.

**How it works:**
- `trainer.yaml` is loaded automatically via `default_config_files` in `__main__.py`
- Stage YAMLs only contain model `class_path` + overrides from `__init__` defaults + data
- Model `__init__` signatures have defaults (sourced from old schema.yaml values) — stage YAML only overrides what differs
- Overlays are optional `--config` adds for scale variants
- `fusion.yaml` overrides trainer settings (precision 32, max_epochs 50, monitors val_acc)

**Usage:**
```bash
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids fit --config graphids/config/stages/autoencoder.yaml --config graphids/config/overlays/small_vgae.yaml
python -m graphids fit --config graphids/config/stages/fusion.yaml
```

**`__main__.py` change needed:**
```python
parser_kwargs={
    "default_config_files": ["graphids/config/trainer.yaml"],
    "default_env": True,
    "env_prefix": "KD_GAT",
}
```

## Remaining Steps: Flatten Models

### Step 1: Flatten VGAEModule (`graphids/core/models/vgae.py`)

**Before:**
```python
from graphids.config.defaults.schema import VGAEConfig, TrainingConfig

def __init__(self, vgae: VGAEConfig = VGAEConfig(), training: TrainingConfig = TrainingConfig(), ...)
    vgae = coerce_config(vgae, VGAEConfig)
    training = coerce_config(training, TrainingConfig)
    self.save_hyperparameters()
```

**After:**
```python
def __init__(
    self,
    # --- architecture ---
    conv_type: str = "gatv2", hidden_dims: list[int] = [480, 240, 48],
    latent_dim: int = 48, heads: int = 4, embedding_dim: int = 32,
    dropout: float = 0.15, edge_dim: int = 11, proj_dim: int = 0,
    variational: bool = True, mask_ratio: float = 0.3,
    k_neg: int = 32, canid_weight: float = 0.1,
    nbr_weight: float = 0.05, kl_weight: float = 0.01,
    # --- training (only fields this module reads) ---
    lr: float = 0.003, weight_decay: float = 0.0001,
    gradient_checkpointing: bool = True, compile_model: bool = False,
    # --- identity / dynamic ---
    model_type: str = "vgae", lake_root: str = "experimentruns",
    dataset: str = "", seed: int = 42, gat_stage: str = "curriculum",
    auxiliaries: list | None = None,
    num_ids: int = 0, in_channels: int = 0, num_classes: int = 2,
):
    super().__init__()
    if auxiliaries is None:
        auxiliaries = []
    self.save_hyperparameters()
    ...
```

**Changes in body:**
- Delete `from graphids.config import coerce_config` (2 sites: `__init__` + `_build`)
- Delete `from graphids.config.defaults.schema import VGAEConfig, TrainingConfig` (top-level)
- `_build()`: construct `GraphAutoencoderNeighborhood(...)` directly from `self.hparams.*` instead of through `from_config(hp, ...)`
- `forward()`: `self.hparams.vgae.mask_ratio` → `self.hparams.mask_ratio`
- `_task_loss()`: `self.hparams.vgae.k_neg` → `self.hparams.k_neg` (etc. for canid_weight, nbr_weight, kl_weight)
- `configure_optimizers()`: `self.hparams.training.lr` → `self.hparams.lr`

### Step 2: Flatten GATModule (`graphids/core/models/gat.py`)

Same pattern. Flat args from schema.yaml `gat` + `training` sections:
- Architecture: `hidden: int = 48`, `layers: int = 3`, `heads: int = 8`, `dropout: float = 0.2`, `fc_layers: int = 3`, `embedding_dim: int = 16`, `conv_type: str = "gatv2"`, `edge_dim: int = 11`, `pool_aggrs: list[str] = ["mean"]`, `proj_dim: int = 0`
- Training: `lr`, `weight_decay`, `gradient_checkpointing`, `compile_model`, `loss_fn: str = "ce"`, `focal_gamma: float = 2.0`, `loss_weight: float = 10.0`
- Identity/dynamic: same as VGAE

Delete schema import, coerce_config calls, nested hparams access.

### Step 3: Flatten DGIModule (`graphids/core/models/dgi.py`)

Flat args from schema.yaml `dgi` + `training` sections:
- Architecture: `conv_type`, `hidden_dims`, `latent_dim`, `heads`, `embedding_dim`, `dropout`, `edge_dim`, `proj_dim`
- Training: `lr`, `weight_decay`, `gradient_checkpointing`, `compile_model`
- Dynamic: `num_ids`, `in_channels`, `num_classes`

### Step 4: Flatten TemporalLightningModule (`graphids/core/models/temporal.py`)

Prefixed spatial args to avoid collision with temporal args:
- Spatial (GAT): `spatial_hidden`, `spatial_layers`, `spatial_heads`, `spatial_dropout`, `spatial_embedding_dim`, `spatial_conv_type`, `spatial_edge_dim`, `spatial_pool_aggrs`, `spatial_proj_dim`, `spatial_fc_layers`
- Temporal: `window_size`, `stride`, `temporal_hidden`, `temporal_heads`, `temporal_layers`, `freeze_spatial`, `spatial_lr_factor`
- Training + dynamic args

Update `_build_model()` to construct `GATWithJK(...)` directly.

### Step 5: Flatten RLFusionModule (`graphids/core/models/fusion_baselines.py`)

Prefixed args for DQN/bandit collision avoidance:
- DQN: `dqn_hidden`, `dqn_layers`, `dqn_gamma`, `dqn_epsilon`, `dqn_buffer_size`, `dqn_batch_size`, `dqn_target_update`, etc.
- Bandit: `bandit_ucb_alpha`, `bandit_lambda_reg`, `bandit_hidden`, `bandit_layers`, `bandit_buffer_size`, `bandit_batch_size`, etc.
- Fusion: `method`, `episodes`, `max_samples`, `decision_threshold`, `lr`, etc.

Update agent construction to pass flat args directly.

### Step 6: Update `_training.py` — `prepare_kd`

`graphids/core/models/_training.py` lines 201-202: `cfg.vgae.latent_dim` → `cfg.latent_dim`. Same for `tcfg.vgae.latent_dim` → `tcfg.latent_dim`.

No fallback — fail fast with actionable error if old nested format detected.

### Step 7: Keep `from_config` on inner nn.Modules but update to flat keys

Used by `registry.py`, `temporal.py`, `test_integration.py`.

```python
@classmethod
def from_config(cls, cfg, num_ids: int, in_ch: int):
    return cls(
        hidden_dims=list(cfg.hidden_dims),  # was cfg.vgae.hidden_dims
        latent_dim=cfg.latent_dim,           # was cfg.vgae.latent_dim
        ...
        use_checkpointing=cfg.gradient_checkpointing,  # was cfg.training.gradient_checkpointing
    )
```

### Step 8: Wire `__main__.py`

```python
GraphIDSCLI(
    pl.LightningModule, pl.LightningDataModule,
    subclass_mode_model=True, subclass_mode_data=True,
    seed_everything_default=42,
    parser_kwargs={
        "default_config_files": ["graphids/config/trainer.yaml"],
        "default_env": True,
        "env_prefix": "KD_GAT",
    },
)
```

### Step 9: Checkpoint migration script

`scripts/migrate_checkpoints.py` — rewrites `hyper_parameters` in `.ckpt` files from nested to flat format. `--dry-run` flag. Run once.

### Step 10: Fix tests

- `tests/conftest.py`: flat `SimpleNamespace` fixture, no `*Config` imports
- `tests/test_integration.py`: cfg is now flat, works after Step 7
- Delete all `from graphids.config.defaults.schema import` lines

### Step 11: Update rules

- `.claude/rules/config-system.md` — update to reflect new layout and flat-arg model pattern

## Files Modified

| File | Action |
|------|--------|
| `graphids/core/models/vgae.py` | Flatten __init__, delete coerce_config/schema imports, update hparams access |
| `graphids/core/models/gat.py` | Same |
| `graphids/core/models/dgi.py` | Same |
| `graphids/core/models/temporal.py` | Flatten __init__, delete schema import, prefixed spatial args |
| `graphids/core/models/fusion_baselines.py` | Flatten RLFusionModule __init__, prefixed DQN/bandit args |
| `graphids/core/models/_training.py` | `prepare_kd` flat access (2 lines) |
| `graphids/core/models/registry.py` | No change — callers pass `module.hparams` which is now flat |
| `graphids/__main__.py` | Wire `default_config_files: ["graphids/config/trainer.yaml"]` |
| `scripts/migrate_checkpoints.py` | New — checkpoint migration |
| `tests/conftest.py` | Flat base_cfg fixture |
| `tests/test_integration.py` | Update cfg construction |
| `.claude/rules/config-system.md` | Update to reflect new layout |

## Verification

1. **Import check (login node safe):** `python -c "from graphids.core.models.vgae import VGAEModule; print('OK')"`
2. **jsonargparse introspection:** `python -m graphids fit --config graphids/config/stages/autoencoder.yaml --print_config`
3. **Tests via SLURM:** `bash scripts/slurm/run_tests_slurm.sh`
4. **Checkpoint round-trip:** Save a tiny checkpoint, reload with `load_from_checkpoint`, verify hparams are flat primitives

## Execution Order

Steps 1-5 (model files) are independent — can be parallelized.
Step 6 (_training.py) depends on Step 1.
Step 7 (from_config) depends on Steps 1-3.
Step 8 (__main__.py) independent.
Step 9 (migration script) independent.
Step 10 (tests) depends on Steps 1-7.
Step 11 (rules) last.
