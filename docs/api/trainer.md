# Core: Trainer & Callbacks

Pure-PyTorch training loop and callback protocol — Lightning was
removed. Single-GPU only (project targets 1× V100/H100), handles
AMP via `GradScaler`, gradient clipping, AMP-safe scheduler
skipping on inf/nan scale-warmup batches, and a callback lifecycle
using the same hook names as Lightning so OTel + curriculum
callbacks ported over without change.

The three first-party callbacks live alongside the loop:

- `ModelCheckpoint` — atomic best + last ckpt persistence, owns
  the `checkpoints/` subdir convention.
- `EarlyStopping` — flips `trainer.should_stop` at the epoch
  boundary; the loop observes the flag after the scheduler step
  so the current epoch's metrics are logged before exit.
- `TauNormCallback` — Kang ICLR 2020 τ-norm of the GAT classifier
  head at fit-end; loaded from the best ckpt, in-place rescale,
  re-saved.

`MLflowTrainingCallback` (in [`graphids._mlflow`](orchestrate.md))
is registered alongside but lives in the MLflow surface for
discoverability. Checkpoint save/load helpers live in
`graphids.core._ckpt` (private — single ownership for ckpt
schema, paired with `ModelCheckpoint`'s save side).

## `graphids.core.trainer`

::: graphids.core.trainer

## `graphids.core.callbacks`

::: graphids.core.callbacks
