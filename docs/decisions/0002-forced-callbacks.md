# Forced Callbacks: Eliminate Config Fragility

> **Stale reference note (2026-04-06):** `_lightning.py` and `add_lightning_class_args` no longer exist. Forced callbacks are now constructed by `_build_callbacks()` in `graphids/instantiate.py`. `defaults/trainer.yaml` replaced by `configs/_lib/defaults.libsonnet`.

> Status: **IMPLEMENTED** | Created: 2026-03-31 | Implemented: 2026-04-01

## Problem

jsonargparse replaces lists atomically. Any stage YAML defining `trainer.callbacks:`
silently dropped ModelCheckpoint + EarlyStopping from `trainer.yaml`. This caused
curriculum runs to train 300 epochs with **no checkpoint** — weights lost on job exit.

## Solution

Callbacks registered via `parser.add_lightning_class_args(CallbackClass, "namespace")`
are injected **after** config file merging. They live in separate namespaces (`checkpoint.*`,
`early_stopping.*`), not in `trainer.callbacks`. No config file can remove them.

Stage YAMLs override via namespace keys (e.g., `checkpoint.monitor: val_acc`),
not `trainer.callbacks` lists. DeviceStatsMonitor in `trainer.yaml` survives.

## Key files

- `_lightning.py` — `add_lightning_class_args` + defaults
- `defaults/trainer.yaml` — shared defaults (must stay in sync with Python constants)
- Stage YAMLs — namespace overrides instead of `callbacks:` blocks

See `docs/reference/config-architecture.md` (§3 Forced Callbacks, §6 strength S3).
