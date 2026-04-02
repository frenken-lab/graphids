# Per-stage recipe overrides

## Problem

`trainer_overrides` in recipes is a flat global dict applied to ALL stages via
`ConfigResolver.resolve()`. But stages have different data modules with different
parameters. For example, `data.init_args.max_epochs` exists on
`CurriculumDataModule` but not on `FusionDataModule` or `CANBusDataModule`.
Applying it globally crashes stages that don't have the field.

## Current workaround

`resolve.py` has a curriculum-specific auto-sync hack (line ~95):

```python
if cfg.stage == "curriculum" and "trainer.max_epochs" in runtime_overrides:
    key = "data.init_args.max_epochs"
    val = runtime_overrides["trainer.max_epochs"]
    runtime_overrides[key] = val
    audit.append(OverrideRecord(key=key, value=val, source="curriculum_sync"))
```

This handles the one known cross-field constraint but doesn't generalize. Every
new stage-specific field would need another `if cfg.stage == ...` block.

## Design options

### Option A: Per-stage override blocks in recipes

Recipes declare overrides scoped to specific stages:

```yaml
trainer_overrides:
  trainer.max_epochs: 2

stage_overrides:
  curriculum:
    data.init_args.max_epochs: 2
  fusion:
    model.init_args.buffer_size: 10000
```

`expand_recipe_configs()` would propagate `stage_overrides[stage]` into each
`StageConfig.trainer_overrides`. The resolver applies them without any
stage-specific logic.

**Pro**: Explicit, no magic auto-sync. Recipe author controls exactly what each
stage gets.
**Con**: Verbose for the common case (epoch override) where the user expects one
knob to propagate everywhere it's needed.

### Option B: Namespace filtering in the resolver

The resolver inspects the merged YAML to determine which keys exist on the target
stage's data/model init_args, and silently drops overrides that don't match:

```python
# In resolve():
merged = self._merge_yaml_chain(cfg.config_files, {})
valid_data_keys = set((merged.get("data", {}).get("init_args", {}) or {}).keys())
for k, v in list(runtime_overrides.items()):
    if k.startswith("data.init_args.") and k.split(".")[-1] not in valid_data_keys:
        audit.append(OverrideRecord(key=k, value=v, source="filtered_inapplicable"))
        del runtime_overrides[k]
```

Cross-field sync (curriculum max_epochs) stays as explicit auto-inject logic.

**Pro**: Recipes stay simple, one flat dict.
**Con**: Silent filtering can mask typos. Requires YAML merge before filtering,
adding latency. Auto-sync hacks for cross-field constraints remain.

## Recommendation

Option A is more aligned with the project's "fail fast, no silent fallbacks"
principle. The curriculum epoch sync becomes an explicit `stage_overrides` entry
in the recipe rather than resolver magic.
