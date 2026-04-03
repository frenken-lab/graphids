# Per-test-set metrics for evaluation

## Problem

`test_step` in GAT (and likely other models) aggregates `self.test_metrics` across all 6 test dataloaders, ignoring `dataloader_idx`. The logged accuracy/AUC/F1 is a single number pooling known-vehicle/known-attack, unknown-vehicle, unknown-attack, suppress, and masquerade test sets.

This produces misleading aggregates (e.g. val_acc=96% → test_acc=17%) because suppress/masquerade are excluded from training (`EXCLUDED_ATTACK_TYPES`) and unknown-vehicle/unknown-attack sets are out-of-distribution by design.

## Fix

Log per-dataloader metrics in `test_step` using `dataloader_idx` to index into a dict of MetricCollections. Log both per-set and aggregate at `on_test_epoch_end`. The test subdir names are available from `self.trainer.datamodule.test_datasets.keys()`.

## Affected files

- `graphids/core/models/supervised/gat.py` — `test_step`, `on_test_epoch_end`
- Likely same pattern in `fusion_baselines.py`, `temporal.py`, `dgi.py`, `vgae.py`

## Why it matters

Paper claims require per-scenario breakdown (claim 4: loss×curriculum, claim 5: conv type). A single aggregate hides where models succeed vs fail.
