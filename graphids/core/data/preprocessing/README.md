# Graph Preprocessing

This package keeps the graph cache path small and explicit:

- `representations.py` defines the supported graph shapes:
  `snapshot` and `snapshot_sequence`.
- `materialization.py` turns raw CAN rows into staged graph tables.
- `graph_ops.py` adds the fixed edge and topology features consumed by
  the CAN schema.
- `pyg.py` packs staged tables into pre-collated PyG tensors.
- `splits.py` builds leakage-safe train/validation graph indices.

`representation_cfg` is the selection surface. Dataset code passes it to
`build_graph_tables`, which either emits one graph per complete window or
combines consecutive snapshot windows into a sequence graph.
