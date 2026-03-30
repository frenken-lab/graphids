# Wire trainer.yaml + verify every config claim

> Status: **complete** | Created: 2026-03-28 | Completed: 2026-03-29 | Audited: 2026-03-30

## What was broken

`trainer.yaml` existed but was never loaded. All training ran with Lightning bare defaults:
FP32, no callbacks, no early stopping, no gradient clipping. Only `fusion.yaml` worked
because it explicitly sets trainer overrides.

## What was done

### 1. Wired `default_config_files` in `cli.py:26-31`

```python
# CLI_KWARGS in cli.py
parser_kwargs={
    "default_env": True, "env_prefix": "KD_GAT",
    **{sub: {"default_config_files": ["graphids/config/trainer.yaml"]}
       for sub in ("fit", "validate", "test", "predict")},
},
```

Merge order: `trainer.yaml` < `--config stage.yaml` < `--config overlay.yaml` < CLI args.

### 2. trainer.yaml contents (24 lines)

- `precision: 16-mixed`, `max_epochs: 300`, `gradient_clip_val: 1.0`
- Loggers: WandbLogger (`project: kd-gat`) + CSVLogger
- 4 callbacks: ModelCheckpoint (`filename: "best_model"`), EarlyStopping (`patience: 100`),
  CurriculumEpochCallback, DeviceStatsMonitor

### 3. Checkpoint path aligned with orchestration

`component.py:322,406` checks `run_dir / "checkpoints" / "best_model.ckpt"` —
matches Lightning's auto-resolved path with `logger: false` disabled (loggers present,
so path is `{save_dir}/lightning_logs/version_N/checkpoints/` unless stage sets `default_root_dir`).

### 4. fusion.yaml overrides

`precision: 32`, `max_epochs: 50`, own callbacks (ModelCheckpoint with `monitor: val_acc`).
List replacement semantics — no EarlyStopping or CurriculumEpochCallback in fusion.

## Verified (via `--print_config`)

- [x] autoencoder/normal/curriculum: `precision: 16-mixed`, `max_epochs: 300`, 4 callbacks
- [x] fusion: `precision: 32`, `max_epochs: 50`, 1 callback with `val_acc`
- [x] overlay stacking works (overlay wins for model, trainer.yaml wins for trainer)
- [x] `config-system.md` merge order documented

## Deferred to Phase E

- [ ] gpudebug spike produces `checkpoints/best_model.ckpt` in run dir
- [ ] Re-run triggers skip-if-done

## Open concern

- **CurriculumEpochCallback as default callback** — loaded for ALL stages, not just curriculum.
  Verify it's a no-op for non-curriculum stages or move to `curriculum.yaml` only.

Checkpoint dirpath is fine — `component.py:338` passes `--trainer.default_root_dir={rd}` as
CLI arg, so checkpoints land at `{rd}/checkpoints/best_model.ckpt` regardless of logger config.

## Files changed

1. `graphids/cli.py` — `default_config_files` in `CLI_KWARGS`
2. `graphids/config/trainer.yaml` — `filename: "best_model"` on ModelCheckpoint
3. `graphids/orchestrate/component.py` — checkpoint path updated
4. `.claude/rules/config-system.md` — merge order documented
