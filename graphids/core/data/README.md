# Core Data Layer

This package owns the path from raw dataset files to training-facing graph
representations.

The current architecture is documented in:

- [`docs/reference/data-architecture.md`](../../../docs/reference/data-architecture.md)

At a glance:

- `datasets/` adapts raw sources into dataset/source contracts.
- `discovery/` stores raw signal profiles and provisional hypotheses.
- `preprocessing/` turns rows into views, segments, graph tables, PyG
  tensors, and temporal streams.
- `state.py` keeps process-local dataset state in memory for reuse within
  one Python process.

The training path now treats representation config as the primary surface;
window sizes and strides are derived from that config unless explicitly
overridden for compatibility.
