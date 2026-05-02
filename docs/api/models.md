# Core: Models

Model families used as ablation rows. All inherit from
`GraphModuleBase` (`base.py`), which owns the VRAM probe
(`compute_budget`) plus the `_store_init_kwargs` /
`_build_id_encoder` mixins.

- **`autoencoder/`** — VGAE family (unsupervised reconstruction).
  Stage 1 of the KD chain.
- **`supervised/`** — GAT family (supervised classification). Stage 2.
- **`fusion/`** — fusion modules dispatching on `fusion_method` TLA
  over the method libsonnets. Stage 3.
- **`id_encoding/`** — categorical-ID encoders (embedding tables
  with reserved UNK at index 0).

## `graphids.core.models`

::: graphids.core.models
    options:
      show_submodules: true
