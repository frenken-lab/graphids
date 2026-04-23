# Core: Callbacks

Pure-Python callback protocol (``CallbackBase``) with concrete no-op
defaults so subclasses override only what they need. The four
first-party callbacks:

- ``ModelCheckpoint`` — atomic best + last ckpt persistence, owns the
  ``checkpoints/`` subdir convention
- ``EarlyStopping`` — flips ``trainer.should_stop`` at the epoch
  boundary
- ``VRAMDriftCallback`` — one-shot warn when free VRAM drifts past
  threshold (co-resident process / activation-leak detector)
- ``SVDDCalibrationCallback`` — post-fit OCGIN centroid fit for DGI

## `graphids.core.callbacks`

::: graphids.core.callbacks
