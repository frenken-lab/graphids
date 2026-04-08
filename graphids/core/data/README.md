# Data Module

```
data/
  __init__.py              # Public API: CANBusDataset, GraphDataModule, FusionDataModule
  graph_pipeline.py        # Domain-agnostic: sliding window -> PyG graphs (10-step pipeline)
  io.py                    # NFS-safe primitives: atomic_save, nfs_lock, vocab_from_column
  sampler.py               # NodeBudgetBatchSampler, PrefetchLoader
  budget.py                # GPU memory budget computation for dynamic batching
  budget_probe.py          # VRAM probing (measures actual GPU cost per graph)
  cache.py                 # Preprocessed cache management
  fusion_states.py         # Extract VGAE/GAT embeddings for fusion stage input
  schemas.py               # Shared Pydantic schemas
  datamodule/
    graph.py               # GraphDataModule (Lightning) — used by VGAE + GAT stages
    fusion.py              # FusionDataModule (Lightning) — used by fusion stage
  datasets/
    can_bus.py             # CAN bus adapter: taxonomy, feature schema, Polars exprs, dataset
    wadi.py                # WaDi adapter (template)
    epic.py                # EPIC adapter (template)
```

## How it fits together

```
Raw data (CSV/pcap)
    |
    |  Dataset adapter (e.g. can_bus.py)
    |    - defines feature schema (column orders, Polars expressions)
    |    - reads raw files, normalizes columns
    |    - calls sliding_window_graphs() with its schema
    v
graph_pipeline.py :: sliding_window_graphs()
    - domain-agnostic 10-step pipeline
    - windowing -> feature aggregation -> graph structure -> tensor packing
    - outputs (Data, slices) for InMemoryDataset cache
    |
    v
Cached .pt files (data_train.pt, data_test.pt)
    |
    v
DataModule (graph.py or fusion.py)
    - loads cached dataset via dataset_cls string
    - wraps in DataLoader with dynamic batching
    - passed to Lightning Trainer
```

## Adding a new dataset

See `datasets/README.md` for the step-by-step guide and `datasets/wadi.py` / `datasets/epic.py`
for starter templates.
