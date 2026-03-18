# KD-GAT Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**. Adding a new value along any axis = adding a YAML file.

## Architecture (3 files)

| File | Role |
|------|------|
| `handler.py` | `ConfigHandler` class — YAML loading, resolution, path derivation. `EnvironmentSettings` (pydantic-settings) for `KD_GAT_*` env vars. |
| `schema.py` | All Pydantic models — `PipelineConfig`, architecture sub-configs, `DatasetEntry`, artifact contracts (`TrainingArtifact`, etc.). |
| `__init__.py` | Singleton + re-exports. All external code: `from graphids.config import X`. |

Config is **inert** — no mlflow, shutil, or I/O. Artifact management lives in `pipeline/artifacts.py`.

## Pipeline topology

`config/pipeline.yaml` is the single source of truth for model types, scales, stages, variants, DAG dependencies, preprocessing constants, defaults, and path defaults. `ConfigHandler` loads this once and exposes `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`. To add a new model/stage/variant, edit `pipeline.yaml` + register the implementation.

## Resolution order

Pydantic defaults (baseline) → `models/{type}/{scale}.yaml` (overrides only) → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

```python
from graphids.config import resolve, PipelineConfig
cfg = resolve("vgae", "large", dataset="hcrl_sa")          # No KD
cfg = resolve("gat", "small", auxiliaries="kd_standard")    # With KD
cfg.vgae.latent_dim    # Nested sub-config access
cfg.training.lr        # Training hyperparameters
cfg.has_kd             # Property: any KD auxiliary?
cfg.kd.temperature     # KD auxiliary config (via property)
cfg.active_arch        # Architecture config for active model_type
```

## Environment variables

All `KD_GAT_*` env vars are declared in `EnvironmentSettings(BaseSettings)` with `env_prefix="KD_GAT_"`. pydantic-settings handles type validation and override priority (env var > YAML default).

## Path layout

`experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]/seed_{N}`

## Config Discipline

- **Version-gate at the boundary, not inline.** When adding optional fields to data objects, add a single normalization step at load time that fills defaults. Don't scatter `hasattr` checks through business logic.
- **Always use `from_config()`**. ALL model construction sites must use `Model.from_config(cfg, ...)` — never manual `Model(param1=...)`. Manual construction silently diverges. `from_config()` must include all training-time params.
- **YAML holds all data, Python just loads it.** Constants, defaults, and infrastructure values live in `pipeline.yaml` / `resources.yaml`. Python code reads and exposes, never hardcodes.
