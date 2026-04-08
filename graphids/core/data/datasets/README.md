# Adding a Dataset Adapter

A dataset adapter connects a raw data source to the generic `graph_pipeline.py`.
You provide the domain knowledge; the pipeline handles windowing, graph construction,
and tensor packing.

## What you need to define

### 1. Node identity

What are your graph nodes? Each row in your DataFrame must have a `node_id` column
(Int64) identifying which node that measurement belongs to.

| Domain | Nodes | node_id source |
|--------|-------|----------------|
| CAN bus | Arbitration IDs | hex arb_id -> dense int via vocab |
| ICS (WaDi/EPIC) | Sensors/actuators | melt wide CSV -> column name -> dense int |

### 2. Feature schema

Define the tensor column layout — what features each node/edge carries:

```python
NODE_COL_ORDER = ["feat_mean", "feat_std", "feat_range", ...]  # determines x.shape[1]
EDGE_COL_ORDER = ("iat", "correlation", ...)                    # determines edge_attr.shape[1]
```

### 3. Polars expressions

These tell the pipeline *how* to compute features from raw data within each window:

```python
# Per-node stats: group_by([_wid, node_id]).agg(NODE_STAT_EXPRS)
NODE_STAT_EXPRS = [
    pl.col("value").mean().alias("feat_mean"),
    pl.col("value").std().alias("feat_std"),
    ...
    pl.lit(0.0).alias("clustering_coeff"),   # placeholder — pipeline fills from graph structure
    pl.lit(0.0).alias("in_degree"),          # placeholder
    pl.lit(0.0).alias("out_degree"),         # placeholder
]

# Per-edge stats: applied after sort([_wid, _row]) with shift-1 adjacency
EDGE_STAT_EXPRS = [
    pl.col("timestamp").diff().over("_wid").cast(pl.Float32).alias("iat"),
    ...
]

# Per-window labels: first must be aliased "y"
LABEL_EXPRS = [
    (pl.col("attack").max() > 0).cast(pl.Int64).alias("y"),
]
```

**Important:** `clustering_coeff`, `in_degree`, `out_degree` must appear as `pl.lit(0.0)`
placeholders in `NODE_STAT_EXPRS`. The pipeline overwrites them with values computed
from the actual graph structure (triangle counting + degree counts in Polars).

### 4. Dataset class

Subclass `InMemoryDataset` and implement:

- `_read_raw()` — load CSVs, normalize columns, return a DataFrame with
  `timestamp`, `node_id`, and whatever your expressions reference
- `_build_graphs()` — build vocab if needed, call `sliding_window_graphs()`
  with your schema constants
- `process()` — NFS-locked cache build (copy the pattern from `can_bus.py`)

### 5. Wire it up

1. Add your class to `datasets/__init__.py`
2. Add a registry entry in `configs/datasets/dataset_registry.json`
3. Use it: `python -m graphids fit --config ... --tla 'dataset="your_dataset"'`
   with `dataset_cls` pointing to your adapter

## The pipeline contract

`sliding_window_graphs(df, window_size, stride, *, exprs...)` expects:

| Column | Type | Required by |
|--------|------|-------------|
| `node_id` | Int64 | Windowing + node aggregation |
| `timestamp` | Float64 | Edge IAT + window ordering |
| `_first_half` | Bool | Created by pipeline, used if your NODE_STAT_EXPRS reference it |
| `attack` | Int64 | Only if your LABEL_EXPRS reference it |
| *(your feature cols)* | Float32 | Whatever your expressions reference |

Returns `(Data, slices, num_graphs)` — the pre-collated InMemoryDataset format.

## ICS datasets: wide-to-long conversion

CAN bus data is naturally long-format (one row per message per CAN ID). ICS datasets
like WaDi and EPIC are wide-format (one row per timestep, one column per sensor).

To reuse the pipeline, melt wide to long:

```python
# Wide: [timestamp, sensor_1, sensor_2, ..., attack]
# Long: [timestamp, node_id, value, attack]
lf = lf.unpivot(
    index=["timestamp", "attack"],
    on=sensor_columns,
    variable_name="sensor_name",
    value_name="value",
)
```

Then `node_id` = dense integer per sensor name (via `vocab_from_column`).

## Reference implementation

`can_bus.py` is the complete reference. It defines:
- `ATTACK_TYPE_CODES` — domain taxonomy
- `BYTE_COLS`, `NODE_COL_ORDER`, `EDGE_COL_ORDER` — feature schema
- `NODE_STAT_EXPRS`, `EDGE_STAT_EXPRS`, `LABEL_EXPRS` — Polars expressions
- `parse_payload()` — domain-specific transform (hex -> bytes)
- `CANBusDataset` — the InMemoryDataset adapter
