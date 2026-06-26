# CAN bus view primitive draft

The current CAN path is snapshot-window based: one graph per sliding window.
That is still a good default, but it is not the only useful lens.

## Current sources

These are the files that define the live snapshot pipeline today:

- `graphids/graphids/core/data/datasets/can_bus.py`
  - CAN-specific schema, raw CSV normalization, payload parsing, attack-type inference.
- `graphids/graphids/core/data/datasets/_base.py`
  - shared dataset/cache control plane, metadata merge, vocab/scaler wiring, split handling.
- `graphids/graphids/core/data/preprocessing/representations.py`
  - supported graph representation configs.
- `graphids/graphids/core/data/preprocessing/materialization.py`
  - sliding-window and snapshot-sequence graph table construction.
- `graphids/graphids/core/data/preprocessing/pyg.py`
  - staged graph-table tensor packing.
- `graphids/graphids/core/data/preprocessing/graph_ops.py`
  - default graph transforms applied after window aggregation.
- `graphids/graphids/core/data/preprocessing/vocab.py`
  - shared vocabulary scan/persist/load primitives.
- `graphids/graphids/core/data/datamodule/graph.py`
  - Lightning datamodule that consumes the processed snapshot graphs.

Drafted views:

- `snapshot`
  - One graph per fixed window.
  - Current pipeline shape.
  - Best when you want the simplest batching and the most stable feature stats.

- `snapshot_sequence`
  - A short ordered list of snapshot graphs.
  - Best next step when a single snapshot is not predictive enough.
  - Lets the model see attack buildup without jumping to a fully streaming setup.

- `event_chunk`
  - Chunk by message count or duration instead of sliding window stats.
  - Useful when the raw event cadence matters more than fixed-length snapshots.

- `rolling_stream`
  - Online view with bounded history.
  - Useful for low-latency or incremental detection.

Likely missing view families:

- `multi_scale_snapshot`
  - Same raw source, multiple window sizes.
  - Good when attacks unfold at different temporal scales.

- `cumulative_history`
  - Each sample carries the current window plus a longer trailing history.
  - Good when the state leading into a window matters more than the window alone.

- `contrastive_pair`
  - Pair two aligned windows or two nearby windows for representation learning.
  - Good for self-supervised pretraining and robust anomaly scoring.

- `entity_view`
  - Re-center the graph around a node or arbitration ID and treat the rest as context.
  - Good for attribution and interpretability.

- `cooccurrence_graph`
  - Build a graph over IDs/messages rather than over time windows.
  - Useful if you want relationship structure rather than temporal snapshots.

What the data can support beyond classification:

- sequence prediction
- next-window forecasting
- change-point detection
- anomaly localization
- contrastive/self-supervised pretraining
- reconstruction / denoising
- attribution and explanation
- retrieval of similar attack segments

The next architectural step that seems worth carrying through code is a
canonical entity layer:

- decode vehicle-specific signals locally, behind private DBCs
- map those decoded names into a shared registry of canonical entities
- store the normalized output as a long feature table keyed by
  `canonical_id`, `vehicle_id`, and time
- layer snapshot, snapshot-sequence, and multi-scale views on top of that

That gives you a path toward a feature-store-style substrate without
committing to a distributed backend before the ontology is stable.

The key design idea is to keep the raw adapter stable and make the view a first-class config.
That lets the same CAN logs feed multiple experiments without duplicating the schema code.
