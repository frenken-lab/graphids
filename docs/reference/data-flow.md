# Pipeline Data Flow

The live training data flow is temporal:

```text
raw CAN CSV rows
  -> load and normalize CAN columns
  -> parse payload bytes and entropy
  -> map arbitration IDs through the configured vocabulary policy
  -> build one temporal event per row
  -> add chronological train/validation/test split masks
  -> pack PyG TemporalData tensors
  -> serve batches with TemporalDataLoader
  -> train temporal Lightning module through Ray Train
```

Key invariants:

- `representation_cfg.kind` is `temporal`.
- Batch size means events per batch.
- `stream_id` identifies chronological continuity.
- `reset_after` marks stream or slice boundaries for stateful models.
- `is_warmup` and `is_scored` separate state warmup from metric accounting.
- Unknown IDs are explicit: `0` is `UNK`, with unknown flags and hash buckets in
  event features.

Historical windowed graph materialization, graph-size budgeting, and
snapshot-sequence split embargoes are no longer part of the primary training
path.

