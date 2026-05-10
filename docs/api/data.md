# Core: Data

All dataset, preprocessing, representation, and discovery machinery.

## Current layout

- **`datasets/`** - raw source adapters and dataset/source builders.
- **`discovery/`** - signal profiles, canonical entities, and provisional
  hypotheses for cross-vehicle alignment.
- **`preprocessing/`** - explicit representations, views, segments,
  materialization, PyG packing, temporal streams, scaler config, vocab
  config, and graph transforms.
- **`datamodule/`** - training-time loaders and batching policy.
- **`state.py`** - process-local dataset state for reuse within one Python
  process.

## What to read next

- [`docs/reference/data-architecture.md`](../reference/data-architecture.md)
- [`graphids/core/data/preprocessing`](../../graphids/core/data/preprocessing)
- [`graphids/core/data/discovery`](../../graphids/core/data/discovery)

## `graphids.core.data`

::: graphids.core.data
    options:
      show_submodules: true
