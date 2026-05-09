# Preprocessing Pipeline

`GraphPipeline` in [`pipeline.py`](pipeline.py) is the core cache-build
pipeline for graph datasets. It converts a dataframe of timestamped rows
into graph tables and, optionally, pre-collated PyG tensors.

## What it does

- assigns rows to sliding windows
- aggregates node statistics and window labels
- generates edges with the configured `EdgePolicy`
- applies graph transforms such as bidirectional edges and topology
  placeholders
- remaps node IDs to local graph IDs
- writes debug artifacts when `debug_artifacts_dir` is set
- returns either staged tables or pre-collated tensors

## Key types

- `GraphPipeline` orchestrates the full flow
- `EdgePolicy` defines how source and destination IDs are chosen
- `GraphTransform` mutates node and edge tables after edge generation
- `TOPOLOGY_NODE_PLACEHOLDER_EXPRS` reserves columns that are filled
  later by topology-aware transforms

## What it does not do

- no model inference
- no batching policy for training
- no dataset-specific parsing

Those concerns stay in dataset adapters or the runtime datamodule.

## Inputs and outputs

- Input: a normalized Polars `DataFrame` with `timestamp`, `node_id`,
  and whatever raw columns the feature expressions need.
- Output: `GraphTables` / `AggregatedTables`, or `Data` + `slices` when
  `run()` is used.

## See also

- [`graphids/core/data/datasets/README.md`](../datasets/README.md)
- [`graphids/core/data/preprocessing/transforms.py`](transforms.py)
- [`docs/reference/data-flow.md`](../../../../docs/reference/data-flow.md)
