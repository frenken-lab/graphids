# Config Flatten — Complete Reference

> Completed: 2026-03-28

Replaced Hydra/OmegaConf + config dataclasses with jsonargparse + flat YAML.
Every LightningModule `__init__` takes flat typed primitives. YAML is law.

## What changed

### Config package (`graphids/config/`)

Deleted: `schema.py` (all `*Config` dataclasses), `coerce_config`, `resolve()`, `constants.py`, `defaults/` directory, `config.yaml`, `models.yaml`, `base.yaml`.

Kept: `__init__.py` (constants + topology + path helpers), `pipeline.yaml`, `datasets.yaml`, `resources.yaml`.

Added: `constants.yaml`, `trainer.yaml`, `stages/*.yaml` (4 files), `overlays/*.yaml` (3 files).

### Model files (5 LightningModules)

| Module | File | Args | Prefix |
|--------|------|------|--------|
| `VGAEModule` | `vgae.py` | 20 | — |
| `GATModule` | `gat.py` | 17 | — |
| `DGIModule` | `dgi.py` | 13 | — |
| `TemporalLightningModule` | `temporal.py` | 20 | `spatial_*`, `temporal_*` |
| `RLFusionModule` | `fusion_baselines.py` | 28 | `dqn_*`, `bandit_*` |

All `*Config` imports, `coerce_config` calls, and nested hparams access (`self.hparams.vgae.*`, `self.hparams.training.*`) eliminated.

### Inner nn.Module `from_config` methods

Updated to flat keys: `GraphAutoencoderNeighborhood`, `GATWithJK`, `GraphInfomaxModel`, `EnhancedDQNFusionAgent`, `NeuralLinUCBAgent`, `reward_kwargs_from_cfg`.

**Exception:** `QNetwork.from_config` — called from `registry._dqn_from_config` with pipeline-constructed nested config. Separate code path, not broken.

### Supporting files

- `_training.py`: `cfg.vgae.latent_dim` → `cfg.latent_dim` in `prepare_kd`
- `submit.py`: `from graphids.config.constants` → `from graphids.config`
- `__main__.py`: stale `TrainingConfig` comment cleaned

### Tests

- `conftest.py`: flat `SimpleNamespace` fixture, zero `*Config` imports
- `test_smoke.py`, `test_vgae.py`, `test_gat.py`: flat kwargs for module instantiation
- `test_features.py`: `EDGE_FEATURE_COUNT` → `N_EDGE_FEATURES` from `features.py`
- `test_config.py`: deleted 6 dead `resolve()` tests

### Checkpoint migration

`scripts/migrate_checkpoints.py` — rewrites `hyper_parameters` in `.ckpt` files from nested to flat. Supports `--dry-run`. Run once against `experimentruns/` before training with new code.

```bash
python scripts/migrate_checkpoints.py experimentruns/ --dry-run
python scripts/migrate_checkpoints.py experimentruns/
```

## Non-Lightning nested references (not broken)

These use pipeline-constructed nested namespaces internally, not LightningCLI:

- `curriculum.py:CurriculumSampler` — `CurriculumDataModule.setup()` builds a synthetic `SimpleNamespace(training=...)`
- `generate.py` — `cfg.evaluation.*` from pipeline runner
- `dqn.py:QNetwork.from_config` — from `registry._dqn_from_config`

Optional cleanup, not correctness issues.

## Verification

1. All 18 modified files pass `ast.parse()` ✓
2. Import check: `python -c "from graphids.core.models.vgae import VGAEModule; print('OK')"`
3. Introspection: `python -m graphids fit --config graphids/config/stages/autoencoder.yaml --print_config`
4. Tests: `bash scripts/slurm/run_tests_slurm.sh`
5. Checkpoint round-trip: save → `load_from_checkpoint` → verify flat hparams
