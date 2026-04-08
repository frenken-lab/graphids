# Pydantic v2 Audit: GraphIDS Codebase

Project pins `pydantic>=2.10` and `pydantic-settings>=2.0` (pyproject.toml:21-22).

## Section 1: Current Usage

Features actively used across 10 files with `from pydantic` imports:

| Feature | Where |
|---------|-------|
| `BaseModel` + `ConfigDict` | All 10 Pydantic files |
| `frozen=True` | `PathContext`, `KDEntry`, `TrainingRunConfig`, `PipelineConfig`, `SweepConfig` |
| `extra="forbid"` | `ValidatedConfig`, `TrainingSpec`, `AnalysisSpec`, `RunRecord`, `_RecipeEnvelope`, `_SweepSpec`, `_SelectionSpec`, `ClassPathBlock`, `_MonitorBlock` |
| `extra="allow"` | `TrainerSection`, `CallbacksSection` |
| `Field(default_factory=...)` | `contracts.py:20`, `orchestrate/contracts/__init__.py:57-62`, `recipes.py:158-175` |
| `Field(min_length=1)` | `ClassPathBlock.class_path` (schemas.py:70), `_MonitorBlock.monitor` (schemas.py:103) |
| `Field(description=...)` | `TrainingSpec.upstream_ckpt_paths` (orchestrate/contracts/__init__.py:58) |
| `model_validator(mode="after")` | `schemas.py:106,141,149,179,194,206`, `cross_field.py:186`, `monarch/schemas.py:37`, `recipes.py:133` |
| `field_validator` | `recipes.py:50,57,64,86,91,98,105,119,126,177` |
| `model_validate()` | `schemas.py:144,150,224` |
| `model_validate_json()` | `core/io.py:82`, `dagster/assets.py:154` |
| `model_dump()` | `recipes.py:141,194`, `planner.py:185`, `cli/_analysis.py:21` |
| `model_dump(mode="json")` | `contracts.py:32` |
| `model_dump_json(indent=2)` | `core/io.py:71` |
| `model_rebuild()` | `schemas.py:124-126` (deferred annotation resolution) |
| `create_model()` | `core/_schema_gen.py:57` (dynamic schema from `__init__` signatures) |
| `Literal` discriminator fields | `RunRecord.status`, `RunRecord.source`, `KDEntry.type`, `cross_field.ValidationRule.severity` |
| `ClassVar` | `TrainingSpec.CONTRACT_NAME/VERSION`, `AnalysisSpec.CONTRACT_NAME/VERSION` |
| `arbitrary_types_allowed` | `cross_field.py:184` (`StageValidation` accepts dataclass fields) |

## Section 2: Ignored Features

### 2a. `@computed_field` for serialization-visible properties

`PathContext` (schemas.py:40-62) has 5 `@property` methods (`run_dir`, `ckpt_file`, `complete_marker`, `last_ckpt_file`, `ckpt_dir`) that are invisible to `model_dump()` and `model_dump_json()`. Same for `ResourceSpec.mem_mb` and `ResourceSpec.time_minutes` (slurm/resources.py:45-55). Using `@computed_field` would include them in serialized output and JSON schema automatically.

### 2b. `pydantic-settings` `BaseSettings` for env var loading

The project has 25+ `os.environ.get("KD_GAT_*")` calls scattered across 12 files (constants.py:69, slurm/env.py:15-16, budget.py:37-46, instantiate.py:24, resources.py:60, staging.py:79-93, paths.py:102,128, fusion_states.py:171, cache.py:51, graph.py:93, vgae_module.py:58, gat_module.py:49, dgi_module.py:37, analyzer.py:35, dagster/definitions.py:32). `pydantic-settings` is already a dependency but never imported. A single `BaseSettings` subclass with `env_prefix="KD_GAT_"` would centralize all env vars with type validation, defaults, and `.env` file support.

### 2c. Discriminated unions

`RunRecord.source` uses `Literal["dagster", "cli"]` but there is no discriminated union dispatch. The callback sections in `CallbacksSection` (schemas.py:128-157) manually parse `checkpoint` and `early_stopping` init_args through separate `model_validate` calls instead of using a discriminated union on `class_path`. Similarly, `_RecipeEnvelope` has `sweeps: list[_SweepSpec]` where sweeps could be discriminated by `model_family` or `stage`.

### 2d. `Annotated` validators for reusable constraints

`recipes.py` has 7 near-identical `@field_validator` methods that check membership in a set (lines 50-131: `_valid_scale`, `_valid_conv_type`, `_valid_loss_fn`, `_valid_fusion_method`, `_valid_model_type`). These repeat the pattern `if v not in VALID_SET: raise ValueError`. A reusable `Annotated[str, AfterValidator(check_in(VALID_SCALES))]` type alias would eliminate the boilerplate.

### 2e. `model_dump(exclude_unset=True)` / `exclude_defaults=True`

`expand_recipe_configs` (recipes.py:194) uses `model_dump(exclude_none=True)`. In `orchestrate/analysis.py:63-64`, `model_dump(mode="python")` is called then `metadata` is popped manually. `model_dump(exclude={"metadata"})` is used at `cli/_analysis.py:21` but inconsistently.

### 2f. Strict mode for config boundaries

No model uses `ConfigDict(strict=True)` or `Field(strict=True)`. The `_MonitorBlock` (schemas.py:106-109) hand-validates `mode in ("min", "max")` instead of using `Literal["min", "max"]` which Pydantic enforces natively.

### 2g. `model_json_schema()` for config documentation

No call to `model_json_schema()` anywhere. The project has complex config schemas (`ValidatedConfig`, `TrainingSpec`, `RunRecord`) that could auto-generate JSON Schema for documentation or editor autocompletion.

### 2h. `from_attributes=True` for dataclass interop

`StageConfig` (planning/shared.py:15-51) and `ResourceSpec` (slurm/resources.py:29-55) are stdlib `@dataclass(frozen=True)` with manual `to_dict()`/`from_dict()` methods. Pydantic's `from_attributes=True` on consuming models would allow direct validation from these dataclasses without manual conversion.

## Section 3: Handrolled Replacements

### 3a. Env var loading (vs `pydantic-settings`)

**Scope:** 25+ `os.environ.get()` calls across 12 files.

`slurm/env.py` (15-38) is a hand-built env var module: module-level `os.environ.get()` with defaults, plus 3 helper functions with manual `int()` coercion and `try/except`. `config/constants.py:69` does the same for `LAKE_ROOT`. `core/data/budget.py:37-46` reads 3 env vars with `float()`/`int()` coercion. All of this is what `BaseSettings` does natively with type validation.

### 3b. Manual `mode` enum validation (vs `Literal`)

`_MonitorBlock._mode_is_min_or_max` (schemas.py:106-109) is a `model_validator` that checks `self.mode not in ("min", "max")`. This is exactly what `mode: Literal["min", "max"]` does at the type level -- Pydantic rejects invalid values automatically.

### 3c. Manual set-membership validators (vs `Literal` or `Annotated`)

`PipelineConfig._validate_axes` (monarch/schemas.py:37-51) manually checks `self.scale not in VALID_SCALES` and `self.fusion_method not in VALID_FUSION_METHODS`. Since these sets are loaded from JSON at import time, `Literal` is not directly usable, but `Annotated[str, AfterValidator(...)]` with a shared validator factory would deduplicate the 10+ validators that all follow this pattern.

### 3d. Contract envelope serialization (vs native Pydantic)

`contracts.py:23-50` hand-builds a versioned envelope system: `to_envelope()` calls `spec.model_dump(mode="json")` and wraps it; `from_envelope()` validates the envelope then calls `spec_cls(**envelope.payload)`. The `CONTRACT_NAME`/`CONTRACT_VERSION` are `ClassVar` attributes accessed via `getattr()`. This could be a discriminated union on `contract` field, with Pydantic handling the dispatch.

### 3e. `StageConfig.to_dict()` / `from_dict()` (vs Pydantic model)

`StageConfig` (planning/shared.py:35-50) is a `@dataclass` with manual `to_dict()` (calls `dataclasses.asdict`) and `from_dict()` (coerces `list` back to `tuple`). As a Pydantic model, `model_dump()` and `model_validate()` handle this natively, including the tuple coercion via `field_validator(mode="before")`.

### 3f. ISO 8601 timestamps as `str` (vs `datetime` types)

`RunRecord` (core/run_record.py:33-34) stores `started_at: str` and `completed_at: str | None` with manual `.isoformat()` calls at write sites (core/models/base.py:412,444). Pydantic's `AwareDatetime` type handles parsing, validation, and ISO 8601 serialization natively.

## Section 4: Recommendations (prioritized)

### P0 -- Quick wins (< 30 min each, immediate value)

1. **`mode: Literal["min", "max"]`** on `_MonitorBlock` (schemas.py:104). Delete the `_mode_is_min_or_max` model_validator entirely. 3 lines deleted.

2. **`@computed_field`** on `PathContext.run_dir` (schemas.py:40-62). Makes `run_dir` visible in `model_dump()` output. Repeat for `ckpt_file`, `complete_marker` etc. if serialization is needed.

### P1 -- Medium effort, high value

3. **`BaseSettings` for `KD_GAT_*` env vars.** Create one `class GraphIDSSettings(BaseSettings)` with `model_config = SettingsConfigDict(env_prefix="KD_GAT_")`. Fields: `lake_root`, `scratch`, `data_root`, `cluster`, `slurm_account`, `slurm_log_dir`, `lake_write`, `dry_run`, `budget_safety_margin`, `budget_grad_mult`, `budget_fallback_bpn`. Replaces 25+ scattered `os.environ.get()` calls. Type coercion (str to float/int/bool) is automatic. `.env` file support built in.

4. **Reusable `Annotated` validator** for set-membership checks. Define `ValidScale = Annotated[str, AfterValidator(lambda v: _check_in(v, VALID_SCALES, "scale"))]` and similar. Replaces 10+ `@field_validator` methods in `recipes.py` and `monarch/schemas.py`.

5. **`AwareDatetime`** for `RunRecord.started_at` / `completed_at`. Pydantic handles ISO 8601 parsing and serialization. Eliminates manual `.isoformat()` calls.

### P2 -- Larger refactors, justified when touching these modules

6. **Promote `StageConfig` to Pydantic `BaseModel`**. Gets `model_dump()`, `model_validate()`, tuple coercion, `extra="forbid"` for free. Eliminates `to_dict()`/`from_dict()`.

7. **Promote `ResourceSpec` to Pydantic `BaseModel`**. Gets `@computed_field` for `mem_mb`/`time_minutes`, validated SLURM time format via `field_validator` (already has `__post_init__`), and `model_dump()` for SLURM script generation. Eliminates `dataclasses.replace()` calls.

8. **`model_json_schema()` export** for `ValidatedConfig`, `TrainingSpec`, `RunRecord`. Generate JSON Schema files for config documentation or editor autocompletion in jsonnet.
