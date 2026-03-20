# Phase D: `hydra.utils.instantiate()` for Callbacks + Scheduler

**Date:** 2026-03-20
**Status:** Ready
**Depends on:** Phase C (done)

---

## Scope (revised from research plan)

The original Phase D plan had three items. Research eliminated one:

| Item | Verdict | Reason |
|------|---------|--------|
| `instantiate()` for callbacks in `trainer_factory.py` | **YES** | Clean fit — callbacks are constructed from config values, no special logic |
| `instantiate()` for scheduler in `build_optimizer_dict` | **YES** | Eliminates 3-way if/elif dispatch |
| `Tuner.scale_batch_size()` for `batch_sizing.py` | **NO** | Incompatible with `DynamicBatchSampler` — Tuner modifies `batch_size` but DBS uses `max_num_nodes`. KD-GAT uses DBS for variable-size graphs. |
| `instantiate()` for `registry.py` | **NO** | Registry does fusion feature extraction (runtime state), not just construction. `_target_` can't replace `extractors()`, `feature_layout()`, `fusion_state_dim()`. |

**Net scope:** 2 changes to `trainer_factory.py` + YAML additions to `config.yaml`. `batch_sizing.py` and `registry.py` stay unchanged.

---

## Change 1: Callback instantiation

### Current code (`trainer_factory.py:260-289`)

```python
def make_trainer(
    cfg: PipelineConfig,
    stage: str,
    extra_callbacks: list | None = None,
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    t = cfg.training
    out = Path.cwd()
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    csv_logger = pl.loggers.CSVLogger(save_dir=".", name="", version="")

    callbacks = [
        ModelCheckpoint(
            dirpath=str(out),
            filename="best_model",
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.save_top_k,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            patience=t.patience,
            mode=t.monitor_mode,
            check_on_train_epoch_end=False,
        ),
        DeviceStatsMonitor(cpu_stats=False),
        RunMetadataCallback(),
    ]

    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    ...
```

### Target code (`trainer_factory.py`)

```python
def make_trainer(
    cfg: PipelineConfig,
    stage: str,
    extra_callbacks: list | None = None,
) -> pl.Trainer:
    """Create a Lightning Trainer with standard callbacks."""
    t = cfg.training
    torch.backends.cudnn.benchmark = t.cudnn_benchmark

    csv_logger = pl.loggers.CSVLogger(save_dir=".", name="", version="")

    callbacks = _instantiate_callbacks(cfg)
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    ...
```

Plus a helper (needed because `instantiate()` returns a dict, Lightning wants a list, and we need to append `RunMetadataCallback` which has no config):

```python
def _instantiate_callbacks(cfg: PipelineConfig) -> list:
    """Instantiate callbacks from config, add non-configurable ones."""
    from hydra.utils import instantiate

    cbs = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]
    cbs.append(RunMetadataCallback())
    return cbs
```

### Exact edits to `trainer_factory.py`

**Edit 1 — Remove unused imports (line 12)**

Old:
```python
from pytorch_lightning.callbacks import DeviceStatsMonitor, EarlyStopping, ModelCheckpoint
```
New:
```python
```
(Delete the line entirely. These classes are now instantiated by Hydra from YAML.)

**Edit 2 — Add `_instantiate_callbacks` function (after line 21, before `_TEACHER_STAGE`)**

Insert:
```python
def _instantiate_callbacks(cfg: PipelineConfig) -> list:
    """Instantiate callbacks from config, add non-configurable ones."""
    from hydra.utils import instantiate

    cbs = [cb for cb in instantiate(cfg.callbacks).values() if cb is not None]
    cbs.append(RunMetadataCallback())
    return cbs
```

**Edit 3 — Replace callback construction in `make_trainer` (lines 271-289)**

Old:
```python
    callbacks = [
        ModelCheckpoint(
            dirpath=str(out),
            filename="best_model",
            monitor=t.monitor_metric,
            mode=t.monitor_mode,
            save_top_k=t.save_top_k,
            save_on_train_epoch_end=False,
        ),
        EarlyStopping(
            monitor=t.monitor_metric,
            patience=t.patience,
            mode=t.monitor_mode,
            check_on_train_epoch_end=False,
        ),
        DeviceStatsMonitor(cpu_stats=False),
        RunMetadataCallback(),
    ]
```
New:
```python
    callbacks = _instantiate_callbacks(cfg)
```

**Edit 4 — Remove `out = Path.cwd()` (line 267)**

Old:
```python
    out = Path.cwd()
```
Delete this line. It was only used as `dirpath=str(out)` which is now in YAML as `dirpath: "."`.

**Edit 5 — Remove `Path` from imports (line 7) IF no other usage remains**

Check: `resolve_teacher_path` returns `Path`, `load_frozen_cfg` uses `Path`, `_load_teacher` uses `Path`. `Path` is still needed. **Do not remove.**

### YAML addition to `config.yaml`

Add a `callbacks` section. Must go **before** the `hydra:` section (which must be last). Add after the `checkpoints:` block.

Insert after line 50 (`temporal: ...`):
```yaml

callbacks:
  checkpoint:
    _target_: pytorch_lightning.callbacks.ModelCheckpoint
    dirpath: "."
    filename: best_model
    monitor: ${training.monitor_metric}
    mode: ${training.monitor_mode}
    save_top_k: ${training.save_top_k}
    save_on_train_epoch_end: false
  early_stopping:
    _target_: pytorch_lightning.callbacks.EarlyStopping
    monitor: ${training.monitor_metric}
    patience: ${training.patience}
    mode: ${training.monitor_mode}
    check_on_train_epoch_end: false
  device_stats:
    _target_: pytorch_lightning.callbacks.DeviceStatsMonitor
    cpu_stats: false
```

### Schema addition to `schema.py`

`PipelineConfig` needs a `callbacks` field. Since `instantiate()` works on DictConfig/dict, this should be a permissive dict:

Add to `PipelineConfig` class (near `checkpoints` field):
```python
    callbacks: dict = Field(default_factory=dict)
```

### `_hydra_bridge.py` update

`_HYDRA_ONLY_KEYS` must include `"callbacks"` so the callbacks dict (which contains `_target_` keys) is stripped before Pydantic validation — OR — the `callbacks` field stays in PipelineConfig as a plain dict and passes through validation.

**Decision:** Keep `callbacks` in PipelineConfig (it's useful for programmatic callers). The dict type is permissive enough to accept the `_target_` entries. No `_HYDRA_ONLY_KEYS` change needed.

BUT: `resolve()` (Compose API path) won't populate `callbacks` from YAML automatically because `resolve()` builds a schema-merge where Pydantic defaults win for missing keys. Check: `resolve()` uses `OmegaConf.merge(schema, hydra_cfg)` — if `hydra_cfg` has `callbacks`, it will override the empty dict default. **This should work without changes.**

---

## Change 2: Scheduler instantiation

### Current code (`trainer_factory.py:227-257`)

```python
def build_optimizer_dict(optimizer, cfg: PipelineConfig):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler:
        return optimizer

    t_max = t.scheduler_t_max if t.scheduler_t_max > 0 else t.max_epochs

    if t.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif t.scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=t.scheduler_step_size,
            gamma=t.scheduler_gamma,
        )
    elif t.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=t.monitor_mode,
            patience=t.patience // 2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    else:
        log.warning("unknown_scheduler_type", scheduler_type=t.scheduler_type)
        return optimizer

    return {"optimizer": optimizer, "lr_scheduler": sched}
```

### Target code

```python
def build_optimizer_dict(optimizer, cfg: PipelineConfig):
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler or not t.scheduler:
        return optimizer

    from hydra.utils import instantiate

    sched = instantiate(t.scheduler, optimizer=optimizer)

    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    return {"optimizer": optimizer, "lr_scheduler": sched}
```

### Exact edits to `trainer_factory.py`

**Edit 6 — Replace `build_optimizer_dict` body (lines 228-257)**

Old (lines 228-257):
```python
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler:
        return optimizer

    t_max = t.scheduler_t_max if t.scheduler_t_max > 0 else t.max_epochs

    if t.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif t.scheduler_type == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=t.scheduler_step_size,
            gamma=t.scheduler_gamma,
        )
    elif t.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=t.monitor_mode,
            patience=t.patience // 2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    else:
        log.warning("unknown_scheduler_type", scheduler_type=t.scheduler_type)
        return optimizer

    return {"optimizer": optimizer, "lr_scheduler": sched}
```

New:
```python
    """Return optimizer or {optimizer, lr_scheduler} dict for Lightning."""
    t = cfg.training
    if not t.use_scheduler or not t.scheduler:
        return optimizer

    from hydra.utils import instantiate

    sched = instantiate(t.scheduler, optimizer=optimizer)

    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": sched, "monitor": t.monitor_metric},
        }
    return {"optimizer": optimizer, "lr_scheduler": sched}
```

### YAML addition to `config.yaml`

The scheduler config goes inside the `training` section of the model YAML files (e.g., `conf/model/vgae_large.yaml`). But since `use_scheduler: false` is the default in schema.py, the scheduler config should go in `config.yaml` as a default that model configs can override.

Add inside the root of `config.yaml` (this will be part of the `training` section when merged):

Actually — the scheduler config needs to live inside `training:` in the schema. Since `training` is a nested Pydantic model (`TrainingConfig`), the `scheduler` field goes there.

### Schema addition to `schema.py`

Add to `TrainingConfig` (near the existing scheduler fields):
```python
    scheduler: dict | None = None
```

Then the existing fields (`scheduler_type`, `scheduler_t_max`, `scheduler_step_size`, `scheduler_gamma`) become dead code. **But** we can't remove them yet because:
1. Model YAML files may reference them
2. The `_target_` approach needs to be verified first

**Decision:** Add `scheduler: dict | None = None` to TrainingConfig. Keep the old fields for now. Mark them for removal in Phase E after verification.

### YAML for scheduler (in `config.yaml`, or in model configs)

Default (no scheduler — matches `use_scheduler: false`):
```yaml
# No scheduler config needed — use_scheduler defaults to false
```

When a model config enables scheduling (e.g., `conf/model/vgae_large.yaml`):
```yaml
training:
  use_scheduler: true
  scheduler:
    _target_: torch.optim.lr_scheduler.CosineAnnealingLR
    T_max: ${training.max_epochs}
```

The `optimizer=` arg is passed programmatically by `build_optimizer_dict`, not from YAML. Hydra `instantiate()` accepts extra kwargs that override/supplement YAML keys.

**Important:** Each model YAML that currently sets `scheduler_type: cosine` (or step/plateau) needs a `scheduler:` block added. Check which model YAMLs set `use_scheduler: true`.

---

## Change 3: Verify which model YAMLs need scheduler blocks

Need to check all `conf/model/*.yaml` files for `use_scheduler` or `scheduler_type` references.

---

## Files changed (summary)

| File | Action | Lines removed | Lines added | Net |
|------|--------|---:|---:|---:|
| `graphids/pipeline/stages/trainer_factory.py` | Edit: remove callback construction, remove scheduler dispatch | -30 | +9 | -21 |
| `graphids/config/conf/config.yaml` | Add: `callbacks` section | 0 | +18 | +18 |
| `graphids/config/schema.py` | Add: `callbacks` + `scheduler` fields | 0 | +2 | +2 |
| `conf/model/*.yaml` (if any use schedulers) | Add: `scheduler:` block | 0 | ~+3 each | ~+3 |
| **Total** | | **-30** | **+29** | **~-1** |

**Phase D is a small net change.** The value is structural: callback config moves from Python to YAML (overridable via Hydra CLI), scheduler dispatch goes from if/elif to declarative.

---

## What NOT to change

- `batch_sizing.py` — stays (Tuner incompatible with DynamicBatchSampler)
- `registry.py` — stays (does more than construction)
- `fusion.py:_make_fusion_trainer()` — stays (fusion baselines use hardcoded simple callbacks; making them configurable adds complexity for no benefit since MLP/WeightedAvg are secondary methods)
- `modules.py` — no changes (only calls `build_optimizer_dict` which keeps same signature)

---

## Execution order

1. Add `callbacks: dict = Field(default_factory=dict)` and `scheduler: dict | None = None` to schema.py
2. Add `callbacks:` YAML block to config.yaml
3. Edit trainer_factory.py: add `_instantiate_callbacks`, replace callback list, replace scheduler dispatch, remove unused import
4. Check model YAMLs for scheduler usage, add `scheduler:` blocks if needed

---

## Verification

After all changes:
- `python -c "from graphids.config import resolve; c = resolve('vgae', 'large'); print(c.callbacks)"` — should print the callbacks dict with `_target_` entries
- `python -c "from graphids.config import resolve; c = resolve('vgae', 'large'); print(c.training.scheduler)"` — should print `None` (default)
- `grep -r "ModelCheckpoint\|EarlyStopping\|DeviceStatsMonitor" graphids/pipeline/stages/trainer_factory.py` — should return zero matches
- `git diff --stat main` — net lines should be negative or near zero
- No new Python files created under `graphids/`
