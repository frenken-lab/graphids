# Forced Callbacks: Eliminate Config Fragility

> Status: **ready** | Created: 2026-03-31 | Predecessor: [trainer-yaml-wiring.md](trainer-yaml-wiring.md)

## Problem

jsonargparse replaces lists atomically. Any stage YAML that defines `trainer.callbacks:`
silently drops every callback from `trainer.yaml`. This caused curriculum runs to train
300 epochs with **no ModelCheckpoint** — weights were lost on job exit.

### Evidence

- `curriculum.yaml:15-17` defined `callbacks: [CurriculumEpochCallback]`, replacing
  `trainer.yaml:17-22` (ModelCheckpoint + EarlyStopping + DeviceStatsMonitor).
- `fusion.yaml:10-12` defines `callbacks: [ModelCheckpoint(val_acc)]`, dropping
  EarlyStopping and DeviceStatsMonitor.
- Expanded configs in `expanded/` show `callbacks: null` for curriculum runs —
  the bug was visible but never validated.
- `trainer-yaml-wiring.md:58-59` flagged CurriculumEpochCallback as an open concern.
  Line 42 documented fusion's list replacement as known. Both led to this failure.

### Root cause

`trainer.yaml` defines critical callbacks as list items. jsonargparse's `--config`
merge replaces lists wholesale (source: jsonargparse `DOCUMENTATION.rst`, "Override
order" section). Any file in the config chain that touches `callbacks:` wipes the
entire baseline. There is no warning.

## Solution: `add_lightning_class_args` (forced callbacks)

LightningCLI has a built-in mechanism for callbacks that must always be present.
Callbacks registered via `parser.add_lightning_class_args(CallbackClass, "namespace")`
are injected **after** config file merging, in `_instantiate_trainer()`:

```python
# lightning/pytorch/cli.py — _instantiate_trainer()
extra_callbacks = [self._get(self.config_init, c)
                   for c in self._parser(self.subcommand).callback_keys]
```

Source: `lightning/pytorch/cli.py`, `_instantiate_trainer()` method. These callbacks are
appended to `trainer.callbacks` after all YAML merging is complete. No config file can
remove them because they live in a separate namespace, not in the `callbacks:` list.

### Why alternatives are worse

| Alternative | Problem |
|-------------|---------|
| `callbacks+:` append syntax | Requires every YAML author to remember `+:`. One omission = silent failure. |
| `before_instantiate_classes` guard (current interim fix) | String-matches `"ModelCheckpoint"` on class paths. Only guards one callback. Fragile. |
| Redeclare full callback list in every stage YAML | Duplicates config. Drift between copies is inevitable. |
| `trainer_defaults` kwarg | Works but less configurable — can't override monitor/mode per stage via YAML. |

## Implementation

### Step 1: Register forced callbacks in `cli.py`

```python
# graphids/cli.py — GraphIDSCLI.add_arguments_to_parser()
def add_arguments_to_parser(self, parser):
    # Existing link_arguments...
    parser.link_arguments("data.init_args.dataset", "model.init_args.dataset")
    parser.link_arguments("data.init_args.lake_root", "model.init_args.lake_root")
    parser.link_arguments("seed_everything", "model.init_args.seed")
    parser.link_arguments("seed_everything", "data.init_args.seed")
    parser.link_arguments("model.init_args.conv_type", "data.init_args.conv_type")
    parser.link_arguments("model.init_args.heads", "data.init_args.heads")

    # Forced callbacks — always present regardless of stage YAML.
    # Configurable via YAML namespaces (e.g. checkpoint.monitor: val_acc)
    # but cannot be removed by list replacement.
    parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
    parser.set_defaults({
        "checkpoint.monitor": "val_loss",
        "checkpoint.mode": "min",
        "checkpoint.save_top_k": 1,
        "checkpoint.save_last": True,
        "checkpoint.filename": "best_model",
    })
    parser.add_lightning_class_args(EarlyStopping, "early_stopping")
    parser.set_defaults({
        "early_stopping.monitor": "val_loss",
        "early_stopping.patience": 100,
        "early_stopping.mode": "min",
    })
```

Imports to add at top of `cli.py`:

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
```

### Step 2: Remove callbacks from `trainer.yaml`

```yaml
# trainer.yaml — callbacks block removed entirely.
# ModelCheckpoint and EarlyStopping are forced via cli.py.
# DeviceStatsMonitor is optional — keep only if desired.
trainer:
  accelerator: auto
  devices: auto
  precision: 16-mixed
  max_epochs: 300
  gradient_clip_val: 1.0
  log_every_n_steps: 50
  logger:
    - class_path: pytorch_lightning.loggers.WandbLogger
      init_args:
        project: kd-gat
        log_model: false
    - class_path: pytorch_lightning.loggers.CSVLogger
      init_args:
        save_dir: .
  callbacks:
    - class_path: pytorch_lightning.callbacks.DeviceStatsMonitor
```

### Step 3: Update `fusion.yaml`

Replace the `callbacks:` block with namespace overrides:

```yaml
# fusion.yaml — before
trainer:
  precision: 32
  max_epochs: 50
  log_every_n_steps: 10
  gradient_clip_val: null
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args: {monitor: val_acc, mode: max, save_top_k: 1, save_last: true, filename: "best_model"}

# fusion.yaml — after
trainer:
  precision: 32
  max_epochs: 50
  log_every_n_steps: 10
  gradient_clip_val: null
checkpoint:
  monitor: val_acc
  mode: max
early_stopping:
  monitor: val_acc
  mode: max
```

No `callbacks:` key at all. Fusion overrides only what differs (monitor metric).
DeviceStatsMonitor from `trainer.yaml` survives because the list is never replaced.

### Step 4: Remove interim `before_instantiate_classes` guard

The forced callback mechanism replaces the manual Namespace injection added in
the interim fix (`cli.py:56-82`). Revert `before_instantiate_classes` to only
handle logger `save_dir` patching and checkpoint `dirpath` patching.

`dirpath` patching stays because it depends on `default_root_dir` which is a
runtime value, not a config default.

```python
def before_instantiate_classes(self):
    """Patch parsed config: logger save_dirs + checkpoint dirpath."""
    if not self.subcommand:
        return
    subcfg = self.config[self.subcommand]
    root_dir = subcfg.trainer.default_root_dir

    # Patch logger save_dirs
    loggers = subcfg.trainer.logger
    if isinstance(loggers, list):
        for lg in loggers:
            if not hasattr(lg, "class_path"):
                continue
            if "WandbLogger" in lg.class_path:
                lg.init_args.save_dir = WANDB_WRITE_DIR
            elif "CSVLogger" in lg.class_path and root_dir:
                lg.init_args.save_dir = root_dir

    if not root_dir:
        return

    # Pin forced ModelCheckpoint dirpath to {default_root_dir}/checkpoints
    if hasattr(subcfg, "checkpoint") and hasattr(subcfg.checkpoint, "dirpath"):
        subcfg.checkpoint.dirpath = f"{root_dir}/{_CKPT_DIR}"
    else:
        subcfg.checkpoint.dirpath = f"{root_dir}/{_CKPT_DIR}"
```

### Step 5: Add validation to `validate.py`

Add a check that the forced callback monitor metrics are actually logged by the model.
Currently `validate_recipe()` (`validate.py:74-80`) only checks logger/LRMonitor
compatibility.

```python
# In validate_recipe(), after parsing:
monitor = cfg.get("checkpoint", {}).get("monitor", "val_loss")
# Verify monitor key exists in model's logged metrics (requires model introspection
# or a static registry of stage -> valid metrics). At minimum, warn if monitor
# doesn't match stage conventions (val_loss for autoencoder/curriculum, val_acc for fusion).
```

### Step 6: Delete dead code

- `CurriculumEpochCallback` in `curriculum.py` — already deleted (moved to
  `CurriculumDataModule.on_train_epoch_start`)
- `curriculum.yaml` `callbacks:` block — already removed
- Remove `CurriculumEpochCallback` references from `trainer-yaml-wiring.md` audit line

## Verification

All verification via `--print_config` (safe on login node):

```bash
cd ~/KD-GAT

# 1. Autoencoder: should show checkpoint(val_loss) + early_stopping(val_loss)
python -m graphids fit --print_config \
  --config graphids/config/stages/autoencoder.yaml 2>&1 | grep -A3 'checkpoint\|early_stopping'

# 2. Curriculum: same as autoencoder (no callback override)
python -m graphids fit --print_config \
  --config graphids/config/stages/curriculum.yaml 2>&1 | grep -A3 'checkpoint\|early_stopping'

# 3. Fusion: should show checkpoint(val_acc) + early_stopping(val_acc)
python -m graphids fit --print_config \
  --config graphids/config/stages/fusion.yaml \
  --config graphids/config/stages/fusion_dqn.yaml 2>&1 | grep -A3 'checkpoint\|early_stopping'

# 4. Overlay stacking: forced callbacks survive overlay
python -m graphids fit --print_config \
  --config graphids/config/stages/autoencoder.yaml \
  --config graphids/config/overlays/small_vgae.yaml 2>&1 | grep -A3 'checkpoint\|early_stopping'

# 5. CLI override: namespace override works
python -m graphids fit --print_config \
  --config graphids/config/stages/autoencoder.yaml \
  --checkpoint.monitor=val_acc 2>&1 | grep -A3 'checkpoint'
```

## Files changed

| File | Change |
|------|--------|
| `graphids/cli.py` | `add_lightning_class_args` for ModelCheckpoint + EarlyStopping; simplify `before_instantiate_classes` |
| `graphids/config/trainer.yaml` | Remove ModelCheckpoint + EarlyStopping from `callbacks:` list |
| `graphids/config/stages/fusion.yaml` | Replace `callbacks:` with `checkpoint:` + `early_stopping:` namespace overrides |
| `graphids/orchestrate/validate.py` | Add monitor metric validation |
| `graphids/core/preprocessing/curriculum.py` | Already done: deleted `CurriculumEpochCallback` |
| `graphids/config/stages/curriculum.yaml` | Already done: removed `callbacks:` block |

## Risk

- **Low**: `add_lightning_class_args` is a documented, stable LightningCLI API.
  Used in Lightning's own examples and tutorials.
- **Migration**: Existing expanded configs in `expanded/` will differ from new ones.
  Not a problem — expanded configs are regenerated per Dagster run.
- **Backward compat**: Anyone using `--trainer.callbacks` CLI override to add callbacks
  still works — forced callbacks are appended separately. But if someone relied on
  `callbacks:` in a YAML to *remove* ModelCheckpoint, that no longer works (which is
  the point).

## After this

- Resubmit the 8 running curriculum jobs (current runs have no checkpoint callback).
- Regenerate expanded configs via Dagster to confirm new structure.
- Spike: run one curriculum + one fusion job on `gpudebug` to confirm checkpoints save.
