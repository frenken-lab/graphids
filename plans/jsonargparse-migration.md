# Plan: Replace Hydra with jsonargparse

## Context

Hydra is unmaintained (last release Feb 2023), hijacks cwd/output dirs, requires `weights_only=False` for checkpoints, and the codebase reimplements its composition anyway. jsonargparse is actively maintained, powers Lightning's CLI, and is a parser not a framework — no side effects.

~50 touchpoints across ~26 files. Split into 4 phases so each phase is independently committable and testable.

## Inventory (what touches Hydra/OmegaConf)

**Core:** `__main__.py` (decorator, ConfigStore, post-hoc merge), `config/__init__.py` (resolve, identity_hash resolver, OmegaConf structured/merge), `config/config.yaml` (oc.env, identity_hash interpolations)

**Pipeline:** `stages/__init__.py` (HydraConfig.get().run.dir, OmegaConf.save), `trainer_factory.py` (hydra.utils.instantiate for callbacks)

**Models (6 files):** `OmegaConf.create(cfg)` in `__init__` to ensure DictConfig, `save_hyperparameters` serializes DictConfig to checkpoint

**Tests (5 files):** `open_dict`, `OmegaConf.create`, `OmegaConf.to_container`

**Deps:** `hydra-core>=1.3.2`, `omegaconf>=2.3`, `hydra-optuna-sweeper>=1.1.2` (installed but unused)

**NOT used:** `--multirun`, Hydra sweeper plugins, Hydra launcher plugins, distributed Hydra

## Phase 1: Decouple config from OmegaConf internals

**Goal:** Config becomes a plain dict/namespace. OmegaConf still used internally by resolve() but not leaked to consumers.

Files: `config/__init__.py`, 6 model files, `datamodule.py`, `stages/__init__.py`

1. `resolve()` returns a plain namespace (or SimpleNamespace) instead of DictConfig
   - Add `OmegaConf.to_object(cfg)` at end of resolve() → converts to plain dict/dataclass
   - Or: keep returning DictConfig for now, but stop relying on DictConfig-specific features

2. Remove `OmegaConf.create(cfg)` guards in model `__init__` methods (6 files)
   - These exist because cfg arrives as dict from checkpoint deserialization
   - Replace with: if isinstance(cfg, dict), convert to SimpleNamespace recursively

3. Replace `open_dict(cfg)` in `populate_config()` with plain attribute setting
   - If cfg is a namespace/dataclass, just `cfg.num_ids = value`

4. Replace `OmegaConf.save(cfg, path)` with `yaml.dump(cfg_to_dict(cfg), path)`

5. Replace `identity_hash` OmegaConf resolver with a plain function
   - `compute_identity_hash(stage, cfg)` → returns `_abcdef01`
   - Called explicitly in `run_stage()` and path construction, not via interpolation

**Checkpoint compatibility:** `save_hyperparameters()` will serialize plain dicts instead of DictConfig. Old checkpoints still contain DictConfig — keep `weights_only=False` for now (Phase 4 handles migration).

## Phase 2: Replace resolve() and __main__.py

**Goal:** No more Hydra. Config loading uses jsonargparse or plain YAML + dataclass merge.

Files: `config/__init__.py`, `__main__.py`, `config/config.yaml`

1. New `resolve()` using jsonargparse:
   ```python
   def resolve(*overrides: str):
       parser = ArgumentParser()
       parser.add_dataclass_arguments(Config, "cfg")
       # Load base YAML, apply preset, apply overrides
       args = parser.parse_args(overrides)
       return args.cfg
   ```
   Or simpler: plain YAML load + dataclass defaults + dict merge. No framework needed for this.

2. New `__main__.py`:
   - Subcommand `run`: single stage execution (replaces `@hydra.main`)
   - Subcommand `manifest`: DAG submission (unchanged)
   - jsonargparse handles CLI parsing, config file loading, `--config` stacking
   - Working directory managed explicitly (mkdir + chdir), not by Hydra

3. Replace `oc.env` resolver:
   - `lake_root` default: `os.environ.get("KD_GAT_LAKE_ROOT", "experimentruns")`
   - `production`: `os.environ.get("KD_GAT_PRODUCTION", "false")`
   - These become Python-side defaults, not YAML interpolation

4. Replace `hydra.utils.instantiate(cfg.callbacks)` in `make_trainer()`:
   - Callbacks defined in Python (already have the class + args in config)
   - Direct construction: `ModelCheckpoint(dirpath=".", monitor=cfg.training.monitor_metric, ...)`
   - Or use jsonargparse's class instantiation

5. Replace `HydraConfig.get().run.dir` in `run_stage()`:
   - Compute run_dir from cfg fields directly (already have the path template)
   - `run_dir = Path(cfg.lake_root) / tier / dataset / f"{model}_{scale}_{stage}_{hash}" / f"seed_{seed}"`

6. Delete `config.yaml` interpolations — paths computed in Python

## Phase 3: Clean up config.yaml and models.yaml

**Goal:** Single YAML file with all defaults. Presets are separate YAML files loaded via `--config`.

1. Merge `config.yaml` infrastructure into the dataclass defaults or a single `defaults.yaml`
2. Convert `models.yaml` presets to individual files: `presets/vgae_small.yaml`, `presets/gat_large.yaml`
   - Used as: `python -m graphids run --config presets/vgae_small.yaml stage=autoencoder`
   - Or: `resolve("--config", "presets/vgae_small.yaml", "stage=autoencoder")`
3. Delete old `config.yaml` and `models.yaml`

## Phase 4: Remove Hydra/OmegaConf dependency

**Goal:** `pip uninstall hydra-core omegaconf hydra-optuna-sweeper`

1. Remove from `pyproject.toml`
2. Add `jsonargparse[signatures]` to dependencies
3. Fix all remaining OmegaConf imports in tests
4. Handle old checkpoint compatibility:
   - `load_from_checkpoint` with `weights_only=False` still works (torch handles pickle)
   - New checkpoints save plain dicts (no DictConfig)
   - No migration needed — old checkpoints just work, new ones are cleaner

## Files changed per phase

### Phase 1 (decouple)
| File | Change |
|------|--------|
| `config/__init__.py` | resolve() returns plain object, identity_hash becomes function |
| `core/models/vgae.py` | Remove OmegaConf.create guard |
| `core/models/gat.py` | Same |
| `core/models/dgi.py` | Same |
| `core/models/temporal.py` | Same |
| `core/models/fusion_baselines.py` | Same |
| `core/preprocessing/datamodule.py` | Remove open_dict, plain attr set |
| `pipeline/stages/__init__.py` | Replace OmegaConf.save, compute run_dir |

### Phase 2 (replace entry point)
| File | Change |
|------|--------|
| `__main__.py` | jsonargparse CLI, delete @hydra.main |
| `config/__init__.py` | jsonargparse-based resolve() |
| `pipeline/stages/trainer_factory.py` | Direct callback construction |

### Phase 3 (clean config files)
| File | Change |
|------|--------|
| `config/config.yaml` | DELETE |
| `config/models.yaml` | DELETE |
| `config/presets/` (new) | Individual preset YAMLs |

### Phase 4 (remove deps)
| File | Change |
|------|--------|
| `pyproject.toml` | Remove hydra-core, omegaconf; add jsonargparse |
| 5 test files | Replace OmegaConf utilities |
| `__main__.py` | Remove add_safe_globals for OmegaConf |

## What stays unchanged
- `pipeline.yaml`, `resources.yaml`, `datasets.yaml`, `constants.py`
- All `cfg.vgae.*`, `cfg.gat.*` field access patterns
- `save_hyperparameters()` calls (Lightning handles dict serialization)
- Manifest builder / DAG submission logic
- All model/training/evaluation code

## Risk mitigation
- Phase 1 is backwards-compatible — OmegaConf still installed, just not leaked
- Each phase is a separate commit, testable independently
- Old checkpoints work with new code (torch.load handles both DictConfig and dict)
- `--multirun` not used, no sweep plugin dependency

## Verification per phase
1. `python -c "from graphids.config import resolve; cfg = resolve(); print(type(cfg), cfg.vgae.latent_dim)"`
2. `python -m graphids run stage=autoencoder model_type=vgae scale=small dataset=hcrl_ch` (smoke test)
3. `python -m graphids manifest ablation.yaml --dry-run`
4. `pip uninstall hydra-core omegaconf && python -c "from graphids.config import resolve"`
