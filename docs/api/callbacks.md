# Core: Callbacks

Pure-Python callback protocol (``CallbackBase``) with concrete no-op
defaults so subclasses override only what they need. The three
first-party callbacks:

- ``ModelCheckpoint`` — atomic best + last ckpt persistence, owns the
  ``checkpoints/`` subdir convention
- ``EarlyStopping`` — flips ``trainer.should_stop`` at the epoch
  boundary
- ``VRAMDriftCallback`` — one-shot warn when free VRAM drifts past
  threshold (co-resident process / activation-leak detector)

DGI's OCGIN centroid is fit fresh at ``Trainer.test`` start (not
persisted in state_dict), so it needs no callback.

## `graphids.core.callbacks`

::: graphids.core.callbacks
