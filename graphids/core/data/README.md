# Data Module

```
data/
  __init__.py              # Public API: CANBusDataset, CANBusSource, GraphDataModule, FusionDataModule, DatasetState
  state.py                 # DatasetState protocol + process-level get_or_build cache
  extract.py               # Extract VGAE/GAT/DGI per-graph features into a TensorDict (fusion input)
  preprocessing/
    pipeline.py            # Domain-agnostic sliding-window -> PyG graphs (was graph_pipeline.py)
    metadata.py            # cache_metadata.json schema + per-split merge writer
    vocab.py               # Shared arb_id vocabulary + digest
    scaler.py              # Per-column feature scalers (z_benign, robust_benign)
    curriculum.py          # Difficulty scoring, tier bucketing, epoch-gating callback
  datamodule/
    graph.py               # GraphDataModule (Lightning) — used by VGAE + GAT stages
    curriculum.py          # CurriculumDataModule subclass — tier-bucketed train batches
    fusion.py              # FusionDataModule (Lightning) — serves the extract.py cache
    sampler.py             # NodeBudgetBatchSampler + pack_offline (FFD)
  datasets/
    can_bus.py             # CAN bus adapter: taxonomy, feature schema, Polars exprs, dataset
    wadi.py                # WaDi adapter (template)
    epic.py                # EPIC adapter (template)
```

Layering: `preprocessing/` writes durable cache artifacts (one shot per
build); `datasets/` adapters compose the preprocessing pieces into a
`DatasetState` (protocol in `state.py`); `datamodule/` consumes that
state and owns DataLoader / batching / sampling policy. `extract.py` is
a separate one-shot job (the first row of `configs/plans/fusion.jsonnet`)
that runs trained upstream models over the cache and emits a TensorDict
the fusion DataModule loads.

## How it fits together

```
Raw data (CSV/pcap)
    |
    |  Dataset adapter (e.g. datasets/can_bus.py)
    |    - defines feature schema (column orders, Polars expressions)
    |    - reads raw files, normalizes columns
    |    - calls preprocessing/pipeline.py :: GraphPipeline.run() with its schema
    v
preprocessing/pipeline.py :: GraphPipeline.run()
    - domain-agnostic 10-step pipeline
    - windowing -> feature aggregation -> graph structure -> tensor packing
    - outputs (Data, slices) for InMemoryDataset cache
    |
    v
Cached .pt files (data_train.pt, data_test_<subdir>.pt) + cache_metadata.json
    |
    v
DataModule (datamodule/graph.py | curriculum.py | fusion.py)
    - loads DatasetState via state.get_or_build(source)
    - wraps in DataLoader with dynamic batching (datamodule/sampler.py)
    - passed to Lightning Trainer
```

## Adding a new dataset

See `datasets/README.md` for the step-by-step guide and `datasets/wadi.py` / `datasets/epic.py`
for starter templates.
