# Preprocessing Pipeline

`GraphPipeline` is now a thin composer over explicit primitives.

The relevant layers are:

- `representations.py`
  - public `snapshot` / `snapshot_sequence` / `multi_scale` / `temporal` /
    `entity` configs
  - bridges to views, segments, and temporal specs
- `views.py`
  - user-facing view configs
- `segments.py`
  - window, sequence, multi-scale, and entity segment primitives
- `materialization.py`
  - raw-row to graph-table materialization
- `pyg.py`
  - staged table to PyG tensor packing
- `temporal.py`
  - `TemporalData` stream construction
- `graph_ops.py`
  - reusable graph-table transforms

`representation_cfg` is the primary selection surface. `window_size` and
`stride` remain as compatibility knobs, but they are now derived from the
representation config unless explicitly overridden.

For the package overview, see:

- [`docs/reference/data-architecture.md`](../../../../docs/reference/data-architecture.md)
