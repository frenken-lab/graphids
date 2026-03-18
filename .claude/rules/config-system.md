# KD-GAT Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**. Adding a new value along any axis = adding a YAML file.

## Architecture (5 files)

| File | Role |
|------|------|
| `_hydra_bridge.py` | `resolve()` via Hydra Compose API — config group selection + overrides → `PipelineConfig`. |
| `constants.py` | Project constants, `load_pipeline_yaml()`, topology (`STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`). Leaf dependency. |
| `paths.py` | Path derivation (`stage_dir`, `checkpoint_path`, lake path primitives). `EnvironmentSettings` for SLURM + MLflow only. |
| `schema.py` | All Pydantic models — `PipelineConfig` (Literal-validated `model_type`/`scale`), architecture sub-configs, `DatasetEntry`, artifact contracts. |
| `__init__.py` | Re-exports from all submodules. All external code: `from graphids.config import X`. |

Config is **inert** — no mlflow, shutil, or I/O. Artifact management lives in `pipeline/artifacts.py`.

## Pipeline topology

`config/pipeline.yaml` defines model types, scales, stages, and DAG dependencies. `constants.py` loads this once and exposes `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`. Default stages and variants live in `config/conf/config.yaml` (Hydra root config). To add a new model/stage/variant, edit `pipeline.yaml` + `conf/` YAMLs + register the implementation.

## Resolution order

Hydra defaults list → config groups (`conf/model/`, `conf/auxiliary/`, `conf/dataset/`) → CLI overrides → `OmegaConf.to_object()` → `PipelineConfig.model_validate()` → frozen.

```python
from graphids.config import resolve, PipelineConfig
cfg = resolve("vgae", "large", dataset="hcrl_sa")          # No KD
cfg = resolve("gat", "small", auxiliaries="kd_standard")    # With KD
cfg.vgae.latent_dim    # Nested sub-config access
cfg.training.lr        # Training hyperparameters
cfg.has_kd             # Property: any KD auxiliary?
cfg.kd.temperature     # KD auxiliary config (via property)
cfg.active_arch        # Architecture config for active model_type
cfg.vgae.canid_weight  # VGAE task loss weights (canid=0.1, nbr=0.05, kl=0.01)
```

## Environment variables

Path vars (`lake_root`) flow through Hydra `oc.env` resolvers in `config/conf/config.yaml` → `PipelineConfig` fields. SLURM + MLflow vars use `EnvironmentSettings(BaseSettings)` in `paths.py` with `env_prefix="KD_GAT_"`.

## Path layout

`{lake_root}/{production|dev/user}/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/seed_{N}`

`lake_root` defaults to `experimentruns` when `KD_GAT_LAKE_ROOT` is unset.

## Config Discipline

- **Version-gate at the boundary, not inline.** When adding optional fields to data objects, add a single normalization step at load time that fills defaults. Don't scatter `hasattr` checks through business logic.
- **Always use `from_config()`**. ALL model construction sites must use `Model.from_config(cfg, ...)` — never manual `Model(param1=...)`. Manual construction silently diverges. `from_config()` must include all training-time params.
- **YAML holds all data, Python just loads it.** Constants, defaults, and infrastructure values live in `pipeline.yaml` / `resources.yaml`. Python code reads and exposes, never hardcodes.
