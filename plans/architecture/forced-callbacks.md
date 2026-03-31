# Forced Callbacks: Eliminate Config Fragility

> Status: **ready** | Created: 2026-03-31

## Problem

jsonargparse replaces lists atomically. Any stage YAML that defines `trainer.callbacks:`
silently drops every callback from `trainer.yaml`. This caused curriculum runs to train
300 epochs with **no ModelCheckpoint** — weights were lost on job exit.

Evidence: `curriculum.yaml` defined `callbacks: [CurriculumEpochCallback]`, replacing
ModelCheckpoint + EarlyStopping + DeviceStatsMonitor from `trainer.yaml`.

## Solution: `add_lightning_class_args`

Callbacks registered via `parser.add_lightning_class_args(CallbackClass, "namespace")`
are injected **after** config file merging in `_instantiate_trainer()`. No config file
can remove them — they live in a separate namespace, not in the `callbacks:` list.

### cli.py — register forced callbacks

```python
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# In GraphIDSCLI.add_arguments_to_parser():
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

### trainer.yaml — remove ModelCheckpoint + EarlyStopping from callbacks list

Keep only DeviceStatsMonitor in `callbacks:`. The forced callbacks handle the rest.

### fusion.yaml — namespace overrides instead of callbacks list

```yaml
# Replace callbacks: block with:
checkpoint:
  monitor: val_acc
  mode: max
early_stopping:
  monitor: val_acc
  mode: max
```

No `callbacks:` key. DeviceStatsMonitor from `trainer.yaml` survives.

### cli.py — simplify before_instantiate_classes

Remove interim `before_instantiate_classes` ModelCheckpoint string-match guard.
Keep only logger `save_dir` patching and checkpoint `dirpath` patching (runtime value).

### validate.py — add monitor metric check

Warn if `checkpoint.monitor` doesn't match stage conventions (val_loss for autoencoder/curriculum, val_acc for fusion).

## Files to change

| File | Change |
|------|--------|
| `graphids/cli.py` | `add_lightning_class_args` + simplify `before_instantiate_classes` |
| `graphids/config/trainer.yaml` | Remove ModelCheckpoint + EarlyStopping from `callbacks:` |
| `graphids/config/stages/fusion*.yaml` | Replace `callbacks:` with namespace overrides |
| `graphids/orchestrate/validate.py` | Add monitor metric validation |

## After applying

- Resubmit the 8 curriculum jobs that ran without checkpoint callback
- Spike: one curriculum + one fusion on `gpudebug` to confirm checkpoints save
