# KD-GAT Config System

Config is defined by four orthogonal concerns: **model_type** (architecture), **scale** (capacity), **auxiliaries** (loss modifiers like KD), and **dataset**. Adding a new value along any axis = adding a YAML file.

**Pipeline topology**: `config/pipeline.yaml` is the single source of truth for what model types, scales, stages, variants, and DAG dependencies exist. `STAGES`, `STAGE_DEPENDENCIES`, `VALID_MODEL_TYPES`, `VALID_SCALES`, and default `PipelineConfig.variants` all derive from this file. To add a new model/stage/variant, edit `pipeline.yaml` + register the implementation.

**Resolution order**: Pydantic defaults (baseline) → `models/{type}/{scale}.yaml` (overrides only) → `auxiliaries/{aux}.yaml` → CLI overrides → Pydantic validation → frozen.

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

**Path layout**: `experimentruns/{dataset}/{model_type}_{scale}_{stage}[_{aux}]`

**Legacy config loading**: Old flat JSON config files still load via `PipelineConfig.load()` with automatic migration.

## Config Discipline

- **Version-gate at the boundary, not inline.** When adding optional fields to data objects, add a single normalization step at load time that fills defaults. Don't scatter `hasattr` checks through business logic.
- **Always use `from_config()`**. ALL model construction sites must use `Model.from_config(cfg, ...)` — never manual `Model(param1=...)`. Manual construction silently diverges. `from_config()` must include all training-time params.
