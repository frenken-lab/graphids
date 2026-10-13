# Data Architecture

> Historical event-model architecture. The current path is documented in [Graph Mamba](graph-mamba.md).

This is the current GraphIDS data layout after the temporal refactor.

## 1. Raw storage

Source of truth: immutable CAN/CPS CSV rows under
`$GRAPHIDS_LAKE_ROOT/raw/<dataset>/`.

Typical fields:

- `timestamp`
- `arb_id` or `arbitration_id`
- `payload` or `data_field`
- `attack`
- inferred `attack_type`
- source provenance from the catalog path

Code surface:

- `graphids/core/data/datasets/can_bus.py`
- `configs/data/datasets.json`

## 2. Representation

The live public representation is:

```yaml
representation_cfg:
  kind: temporal
```

`TemporalRepresentationCfg` is the only active representation config. New
experiment configs should not expose `window_size`, `stride`, graph budgets, or
snapshot-sequence knobs.

Code surface:

- `graphids/core/data/preprocessing/representations.py`
- `graphids/primitives_data.py`

## 3. Temporal event table

Raw CAN rows are sorted per stream and converted into one event per row.

The event table carries:

- `event_id`
- `vehicle_id`, `source_dir`, `source_file`, `row_index`
- `timestamp`
- `src_id`, `dst_id`
- `src_raw`, `dst_raw`
- unknown-ID flags and hash buckets
- `stream_id`
- `reset_after`
- payload bytes, byte deltas, inter-arrival time, entropy
- `y`
- `attack_type`

The default `source.val_split: chronological` splits within each `stream_id`.
The set_01 final training configs use `source.val_split: recordings`: whole
training-source recordings are held out by content hash, stratified by attack
family, with duplicate files assigned once. `source.split_seed: 42` fixes the
assignment independently of the model seed. `val_fraction` targets a fraction
of recordings in each family; small families can produce a larger holdout.

Recording holdouts require benign and attack events after validation warmup
and keep attack examples in mixed training. If the available training
recordings cannot support that split, preprocessing fails instead of using
test recordings. Attack-free training retains only attack-free training
recordings while using the same attack-inclusive validation assignments.
The default vocabulary is fitted on the final training assignments only for
this policy. Validation-only and test-only IDs map to the unknown ID.

Validation and test tables carry `split_id`, `is_warmup`, and `is_scored`
masks so metrics can ignore warmup events. Recording holdouts require full
validation streams; `val_prefix_events` is rejected for this policy.

Code surface:

- `graphids/core/data/preprocessing/temporal.py`
- `graphids/core/data/datasets/recording_split.py`

## 4. PyG packing and cache

Temporal event tables are packed as PyG `TemporalData`:

```text
src, dst, t, msg, y, attack_type, stream_id, reset_after, event_id
```

Optional split tensors include `split_id`, `is_warmup`, and `is_scored`.
Non-tensor metadata is kept out of the event store so `TemporalDataLoader` can
slice batches safely.

`CANBusTemporalSource` writes versioned caches under
`$GRAPHIDS_LAKE_ROOT/cache/v<PREPROCESSING_VERSION>/<dataset>/...`.
Recording caches have a separate `_split_recordings_v1_seed_<seed>` suffix
and persist `split_manifest.json` with source hashes, duplicate aliases,
assignments, class counts, and a digest. Existing chronological caches and
throughput artifacts retain their original paths.

Code surface:

- `graphids/core/data/datasets/can_bus.py`
- `graphids/core/data/state.py`

## 5. Data modules

`TemporalDataModule` is the training-facing datamodule. It loads or builds the
temporal cache, exposes `num_ids`, `in_channels`, and `num_classes`, and serves
train/validation/test streams through PyG `TemporalDataLoader`.

Code surface:

- `graphids/core/data/datamodule/temporal.py`

## 6. Discovery and hypotheses

The discovery layer stores signal profiles and provisional canonical mappings.
It is independent of the temporal training surface.

Code surface:

- `graphids/core/data/discovery/hypotheses.py`
