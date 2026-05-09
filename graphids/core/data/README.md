# Core Data Layer

This package owns the path from raw dataset files to cached graph state.
The responsibilities are split on purpose:

- `datasets/` adapts a raw source into the common dataset contract.
- `preprocessing/` turns rows into graph tables and cached tensors.
- `state.py` keeps process-local dataset state in memory for reuse within
  one Python process.

## Current contract

1. Dataset adapters define the domain schema: node identity, feature
   expressions, edge policy, and labels.
2. The preprocessing pipeline windowizes rows, builds graph tables,
   applies graph transforms, and writes the durable cache.
3. Runtime code reads the cached tensors and should not re-implement
   preprocessing logic.

## See also

- [`docs/reference/data-flow.md`](../../../docs/reference/data-flow.md)
- [`docs/reference/write-paths.md`](../../../docs/reference/write-paths.md)
- [`graphids/core/data/datasets/README.md`](datasets/README.md)
- [`graphids/core/data/preprocessing/README.md`](preprocessing/README.md)
