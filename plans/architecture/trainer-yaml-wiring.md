# Wire trainer.yaml + verify every config claim

> Status: **complete** | Date: 2026-03-28 | Completed: 2026-03-29
> Prerequisite for: Phase C (config expansion)

## Context

`trainer.yaml` was created during the config rewrite (2026-03-26) but never loaded.
Every training run since has used Lightning bare defaults: FP32, no callbacks, no
early stopping, no gradient clipping. Only `fusion.yaml` works because it explicitly
sets trainer overrides. The docs (`config-system.md`) claim a merge order that doesn't
exist in code.

## What's broken (5 of 9 claims wrong)

| Claim | Reality |
|---|---|
| `trainer.yaml` is shared defaults | Never loaded |
| `precision: 16-mixed` | FP32 (2x slower, 2x VRAM) |
| `max_epochs: 300` | None (runs forever) |
| `ModelCheckpoint` configured | No checkpoints saved |
| `EarlyStopping(patience=100)` | No early stopping |

What's correct: overlays, `save_hyperparameters`, `link_arguments`, `fusion.yaml`.

## Fix

### Step 1: Wire `default_config_files` in `GraphIDSCLI`

**File:** `graphids/__main__.py:52`

```python
parser_kwargs={
    "default_env": True,
    "env_prefix": "KD_GAT",
    "default_config_files": ["graphids/config/trainer.yaml"],
},
```

jsonargparse merge order: `default_config_files` < `--config` < CLI args.
Stage YAMLs override trainer.yaml. Overlays override stages. CLI wins.

Path is relative to CWD. All training runs launch from project root.

### Step 2: Verify merge order with `--print_config`

For each stage, verify trainer settings resolve correctly:

```bash
python -m graphids fit --config stages/autoencoder.yaml --config overlays/small_vgae.yaml \
  --data.init_args.dataset=hcrl_sa --print_config
```

Check these fields are NOT null:
- `trainer.precision` = `16-mixed` (from trainer.yaml)
- `trainer.max_epochs` = `300` (from trainer.yaml)
- `trainer.gradient_clip_val` = `1.0` (from trainer.yaml)
- `trainer.callbacks` = 3 callbacks (ModelCheckpoint, EarlyStopping, LRMonitor)

Then verify fusion.yaml overrides work:
```bash
python -m graphids fit --config stages/fusion.yaml --print_config
```
- `trainer.precision` = `32` (fusion overrides trainer.yaml)
- `trainer.max_epochs` = `50` (fusion overrides)
- `trainer.callbacks` = 1 callback with `monitor: val_acc` (fusion replaces list)

### Step 3: ModelCheckpoint `dirpath` resolution

`trainer.yaml` does NOT set `dirpath`. Lightning auto-resolves:
- With logger: `{logger.save_dir}/{name}/version_N/checkpoints/`
- Without logger: `{default_root_dir}/checkpoints/`

Dagster checks `run_dir / "best_model.ckpt"`. Two sub-issues:

**3a: Filename.** Default is `epoch=X-step=Y`. Add `filename: "best_model"` to
trainer.yaml's ModelCheckpoint.

**3b: Dirpath.** With no explicit `dirpath` and default CSVLogger, checkpoints go
to `{default_root_dir}/lightning_logs/version_0/checkpoints/best_model.ckpt`.
Dagster expects `{default_root_dir}/best_model.ckpt`.

Options:
- Set `logger: false` in trainer.yaml → checkpoints go to `{default_root_dir}/checkpoints/best_model.ckpt`. Dagster checks `run_dir / "checkpoints" / "best_model.ckpt"`.
- Or update dagster skip-if-done to glob for `**/best_model.ckpt`.

Simplest: disable default logger, update dagster to check `run_dir / "checkpoints" / "best_model.ckpt"`.

### Step 4: Update dagster skip-if-done path

**File:** `graphids/orchestrate/dagster_defs.py:77`

Change: `run_dir / "best_model.ckpt"` → `run_dir / "checkpoints" / "best_model.ckpt"`

### Step 5: Regenerate spike YAML and verify end-to-end

1. Regenerate expanded YAML (now includes callbacks from trainer.yaml)
2. Submit gpudebug spike
3. Verify `best_model.ckpt` exists at expected path
4. Re-run — verify skip-if-done triggers

### Step 6: Update docs

- `config-system.md` — remove aspirational claims, document actual merge order
- `PLAN.md` — update Phase B/C status

## Files changed

1. `graphids/__main__.py` — add `default_config_files` to parser_kwargs
2. `graphids/config/trainer.yaml` — add `filename: "best_model"` to ModelCheckpoint, set `logger: false`
3. `graphids/orchestrate/dagster_defs.py` — update checkpoint path
4. `.claude/rules/config-system.md` — fix docs
5. `graphids/config/expanded/spike_autoencoder.yaml` — regenerate

## Verification checklist

Every item verified with a command, not assumed:

- [x] `--print_config` for autoencoder shows `precision: 16-mixed`, `max_epochs: 300`, 3 callbacks
- [x] `--print_config` for normal shows same
- [x] `--print_config` for curriculum shows same
- [x] `--print_config` for fusion shows `precision: 32`, `max_epochs: 50`, 1 callback with `val_acc`
- [x] `--print_config` for autoencoder+overlay shows overlay wins for model, trainer.yaml wins for trainer
- [x] Spike YAML has callbacks baked in (verified via expand.py — all 54 YAMLs have callbacks)
- [ ] gpudebug spike produces `checkpoints/best_model.ckpt` in run dir (deferred to Phase E)
- [ ] Re-run of spike triggers skip-if-done (deferred to Phase E)
- [x] `config-system.md` matches actual behavior (default_config_files documented)
