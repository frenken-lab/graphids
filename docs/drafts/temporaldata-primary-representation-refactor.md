# TemporalData Primary Representation Refactor

## Position

The project should make PyG `TemporalData` the primary CAN/CPS representation.
Windowed graph materialization (`snapshot`, `snapshot_sequence`, node/edge
budgeting, prebatched graph packing) should become legacy compatibility code and
then be removed from normal training paths.

This is intentionally an aggressive refactor. The current window pipeline made
early experiments easier, but its hidden cost is now structural: each sample is
a different graph, so training has to solve graph-size packing, CUDA budget
probing, sequence-window leakage, and model-shape instability before it can
learn from the temporal stream. CAN traffic is naturally an event stream. The
data layer should expose that stream directly.

## Target Representation

Represent each CAN row, or each derived transition between adjacent CAN rows, as
one temporal event:

```python
TemporalData(
    src=previous_entity_id,
    dst=current_entity_id,
    t=timestamp,
    msg=event_features,
    y=event_label,
    attack_type=attack_type,
    ...
)
```

Recommended first mapping:

- `src`: previous CAN entity in the same vehicle/source stream.
- `dst`: current CAN entity.
- `t`: normalized event timestamp or monotonically increasing row time.
- `msg`: current payload bytes, byte deltas, inter-arrival time, entropy,
  provenance flags, unknown-ID flags, and optional source/vehicle embeddings.
- `y`: event-level attack label.
- `attack_type`: attack class code for metrics and stratified reporting.

Use `TemporalDataLoader` to batch successive events. Batch size becomes an
event count, not a node/edge budget.

## Goals

- Replace windowed graph preprocessing with a temporal event materialization.
- Make temporal stream training the default path for supervised and anomaly
  detection experiments.
- Remove node/edge packing as a load-bearing training concern.
- Replace representation-aware window splitting with chronological stream
  splitting and explicit warmup/evaluation intervals.
- Make unseen-ID handling an explicit modeling policy instead of an accidental
  vocabulary/cache behavior.
- Preserve only a short-lived legacy path for reproducing historical snapshot
  experiments.

## Non-Goals

- Do not keep the window stack as a parallel first-class architecture.
- Do not add another budgeting layer for temporal batches unless event batches
  prove too large for memory.
- Do not use PyG `FeatureStore` / `GraphStore` as the main answer. They may help
  storage and neighbor sampling later, but they do not by themselves solve the
  representation mismatch.

## Limitations and Design Accounting

### Model Rewrite

The current GAT/VGAE models consume `Data`/`Batch` objects with `x`,
`edge_index`, `edge_attr`, and `batch`. `TemporalData` supplies event batches
with `src`, `dst`, `t`, and `msg`. The temporal refactor therefore requires new
model contracts, not just a dataloader swap.

Initial model families should be:

- event MLP/Transformer baseline over `msg` plus ID embeddings
- recurrent event model with hidden state reset at stream boundaries
- TGN-style memory model over `(src, dst, t, msg)`
- anomaly model that scores event surprise from temporal state

Legacy GAT/VGAE may remain only behind `legacy_snapshot` until removed.

### Train/Val/Test Splits

Window split logic currently tries to prevent overlapping base windows. The new
unit is time-ordered events. Splits must be chronological by source stream:

- train interval
- validation warmup interval
- validation scoring interval
- test warmup interval
- test scoring interval

Metrics must exclude warmup unless an experiment explicitly measures cold-start
performance. Source/file boundaries must reset model memory unless continuity is
known to be real.

### Different CAN IDs Across Splits

TemporalData can carry different IDs across train/val/test, but the policy must
be explicit:

- deployment-realistic default: train-only vocabulary plus `UNK`
- known-vehicle default: known valid ID universe from metadata/DBC/allowlist
- research-only transductive mode: all split IDs known up front, labels unused

Unseen IDs must not silently become learned lookup rows without provenance.
Materialization should add:

- `src_is_unknown`
- `dst_is_unknown`
- hash bucket features for unknown IDs
- optional raw ID metadata for audit only

The model should learn behavior from timing, transitions, payload distribution,
and novelty, not only from exact ID lookup.

### Label Granularity

The existing label is effectively window-level in many experiments. Temporal
training needs an event-level label contract. If raw rows have labels, use them.
If only file/segment labels exist, materialize weak labels and record that fact
in metadata. Evaluation should distinguish:

- event-level detection
- segment-level detection
- attack onset detection
- false-positive rate during benign intervals

### State Leakage

Temporal models can leak through hidden memory even when tensors are split
correctly. The datamodule must own reset points:

- new source file
- new vehicle
- train/val/test boundary
- configured gap after attack segment
- any discontinuity in timestamp or row order

Validation/test warmup may update model memory, but must not update trainable
weights or calibration statistics unless explicitly configured.

### Class Imbalance

Event streams can be far more imbalanced than window graphs. The plan should
support:

- sequential evaluation without resampling
- optional train-only attack oversampling by contiguous event spans
- focal/cost-sensitive losses
- metrics by event, by segment, and by source file

Do not let convenience sampling destroy chronological order in validation/test.

### Batching and CUDA Memory

Temporal event batches remove variable graph packing, but memory is not free.
The load-bearing knobs become:

- events per batch
- history length or memory size
- number of sampled neighbors/history events, if using TGN-style sampling
- truncated backpropagation span for recurrent models

This is simpler than graph node/edge packing because the first-order batch unit
is fixed: events.

### FeatureStore / GraphStore

PyG `FeatureStore` and `GraphStore` can become useful after the stream schema is
stable, especially for large cross-vehicle data or temporal neighbor sampling.
They should not be introduced as the first refactor step. A single local
`TemporalData` cache is the simpler target.

## Proposed Architecture

### Data Flow

```text
raw CAN rows
  -> source-aware sort and validation
  -> train/known vocabulary mapping
  -> temporal event table
  -> TemporalData tensors
  -> chronological split views
  -> TemporalDataLoader
  -> temporal model
```

### Event Table Contract

A staged event table should exist before PyG packing:

- `event_id`
- `vehicle_id`
- `source_dir`
- `source_file`
- `row_index`
- `timestamp`
- `src_id`
- `dst_id`
- `src_raw`
- `dst_raw`
- `src_is_unknown`
- `dst_is_unknown`
- `stream_id`
- `reset_after`
- `msg_*` feature columns
- `y`
- `attack_type`

`stream_id` is the unit of chronological continuity. A new `stream_id` should be
created when source file, vehicle, or known continuity changes.

### PyG Packing Contract

Add a temporal packer that returns:

```python
TemporalData(
    src=...,
    dst=...,
    t=...,
    msg=...,
    y=...,
    attack_type=...,
    stream_id=...,
    reset_after=...,
    event_id=...,
)
```

Do not emit `x`, `edge_index`, `edge_attr`, `slices`, or graph-level labels for
the primary representation.

## Refactor Plan

### Phase 1: Define Temporal Representation

- Add `TemporalRepresentationCfg(kind="temporal")`.
- Remove `window_size` and `stride` from the default public config surface.
- Keep `SnapshotRepresentationCfg` and `SnapshotSequenceRepresentationCfg` only
  under a legacy namespace or compatibility module.
- Update config validation so new experiments default to `kind: temporal`.
- Rename helpers whose names imply windows, for example
  `representation_window_defaults`.

Acceptance criteria:

- A new dataset source can resolve a temporal cache key without any window
  fields.
- New experiment configs do not mention `snapshot`, `window_size`, or `stride`.

### Phase 2: Replace Materialization

- Add `graphids/core/data/preprocessing/temporal.py`.
- Move raw CAN row normalization into reusable row-level functions.
- Build temporal event tables directly from sorted rows.
- Compute event features with row/transition expressions instead of
  `group_by_dynamic`.
- Add stream-boundary detection and reset metadata.
- Add a `temporal_to_pyg()` packer that emits `TemporalData`.

Legacy action:

- Mark `materialization.py`, `graph_ops.py`, and graph-table packing in `pyg.py`
  as legacy.
- Stop adding new behavior to `GraphTables`.

Acceptance criteria:

- Train/val/test sources can build `TemporalData` caches.
- Event counts match raw-row/transition expectations.
- No primary build path calls `_snapshot_tables`, `_sequence_tables`, or
  `graph_tables_to_pyg`.

### Phase 3: Replace Splitting

- Replace `split_graph_indices` with temporal split planning.
- Split by `(stream_id, event_id/timestamp)` ranges, not materialized graph
  indices.
- Add explicit warmup and scoring masks:
  - `is_warmup`
  - `is_scored`
  - `split_name`
- Add checks for memory leakage across train/val/test boundaries.

Legacy action:

- Keep old split tests only under `tests/legacy`.
- Remove window embargo from the primary API.

Acceptance criteria:

- Validation/test metrics can exclude warmup.
- No split shares train events with validation/test.
- Stream reset points are visible in every temporal batch.

### Phase 4: Replace Data Modules

- Promote `TemporalDataModule` from thin wrapper to the primary datamodule.
- Add support for:
  - stream reset metadata
  - warmup/scored masks
  - event batch size
  - deterministic chronological validation/test iteration
  - optional train shuffling only at span level, not event-level for stateful
    models
- Deprecate `GraphDataModule` in the main training builder.

Legacy action:

- Move graph node/edge budgeting behind a `legacy_graph` module.
- Remove `dynamic_batching` from primary experiment configs.

Acceptance criteria:

- Training uses `TemporalDataLoader`.
- No primary temporal experiment calls `node_budget`, `pack_offline`, or
  `Batch.from_data_list`.

### Phase 5: Replace Models

- Introduce a temporal model base class with a clear batch contract:
  `forward_temporal(batch, state=None)`.
- Add first baselines:
  - stateless event classifier
  - recurrent temporal classifier
  - memory-based anomaly scorer
- Add state reset handling in training/validation/test steps.
- Preserve old GAT/VGAE only behind legacy experiment names.

Acceptance criteria:

- At least one supervised temporal model trains end to end.
- At least one anomaly-style temporal model scores event streams.
- Test/eval code records event-level and segment-level metrics.

### Phase 6: Replace Extraction and Fusion

- Update feature extraction to emit event-level TensorDicts instead of
  graph-level states.
- Update fusion models to consume temporal state features or aggregate
  event-level predictions into segment-level decisions.
- Remove assumptions that every upstream model returns one row per graph.

Acceptance criteria:

- Fusion can operate on temporal event predictions/features.
- Analysis artifacts include event, segment, attack-type, and source-file
  indices.

### Phase 7: Remove Budgeting From Primary Path

- Delete primary imports from `graphids/core/budget.py`.
- Quarantine `graphids/core/budgeting/` as legacy snapshot support.
- Remove budget probe references from model base classes used by temporal
  models.
- Update docs to state that graph packing is legacy.

Acceptance criteria:

- Temporal training starts without CUDA budget probing.
- Configs do not expose node or edge budgets.
- Budget tests are either removed or explicitly legacy-scoped.

### Phase 8: Rewrite Docs, Tests, and Configs

- Update `docs/reference/data-architecture.md`.
- Update `docs/reference/data-flow.md`.
- Update `docs/api/data.md`.
- Add a decision record: "TemporalData is the primary CAN representation."
- Rename snapshot experiment configs as legacy or remove them.
- Add temporal smoke configs for each dataset family.
- Replace preprocessing tests with temporal event-table invariants.

Acceptance criteria:

- Docs no longer describe windowed graphs as the default.
- CI covers temporal cache build, split planning, datamodule iteration, model
  smoke training, and evaluation masks.

## Directory Impact

### `graphids/core/data/preprocessing`

Primary work:

- add temporal event materialization
- add temporal PyG packing
- replace representation helpers
- replace split planner

Legacy work:

- quarantine `materialization.py`, `graph_ops.py`, and snapshot packing

### `graphids/core/data/datasets`

Primary work:

- make CAN dataset build temporal event streams
- make vocab policy explicit
- carry stream/source provenance through caches

Legacy work:

- isolate `BaseGraphDataset` and graph-specific cache behavior

### `graphids/core/data/datamodule`

Primary work:

- make `TemporalDataModule` production-grade
- retire graph dynamic batching from default builders

Legacy work:

- keep `GraphDataModule` only for old snapshot reproduction

### `graphids/core/models`

Primary work:

- add temporal model base
- add event classifier/recurrent/memory baselines
- rewrite metrics around event and segment outputs

Legacy work:

- leave GAT/VGAE graph models behind explicit legacy config names

### `graphids/core/budgeting`

Primary work:

- remove from active temporal training path

Legacy work:

- keep only until old snapshot experiments are no longer needed

### `graphids/core/data/extract.py` and Fusion

Primary work:

- switch from graph-level feature rows to event/span-level features
- preserve stream/event IDs for aggregation and audit

### `configs`

Primary work:

- temporal configs become default
- remove `window_size`, `stride`, `snapshot_sequence`, and graph budget knobs
  from new configs

Legacy work:

- move old snapshot configs to a legacy namespace or delete after reproduction
  needs expire

### `tests`

Primary work:

- event materialization tests
- stream split tests
- unseen-ID policy tests
- temporal datamodule tests
- temporal model smoke tests
- state reset/warmup metric tests

Legacy work:

- quarantine old window split and prebatch tests

## Migration Strategy

The migration should be branch-like and decisive:

1. Add temporal representation and event cache beside the old path.
2. Make one temporal smoke experiment train and evaluate.
3. Flip default builders/configs to temporal.
4. Move snapshot paths to `legacy_*` names.
5. Remove primary imports of graph budgeting and graph datamodule.
6. Port or delete downstream analysis/fusion assumptions.
7. Delete legacy code after historical reproduction is no longer required.

Avoid a long-lived dual architecture. During the transition, every new feature
must target temporal first. Snapshot fixes should be limited to breakage that
blocks migration or historical comparison.

## Open Decisions

- Should the first temporal event be a self-event (`src == dst`) or a transition
  from previous ID to current ID?
- Is the default vocabulary train-only, known-valid-ID, or configurable per
  dataset?
- What is the canonical stream boundary for datasets whose CSV files are already
  attack-specific?
- How much warmup should validation/test receive before scoring?
- Should anomaly scoring be event-level first, then segment-aggregated, or
  segment-level from the start?
- Which temporal baseline becomes the minimum viable replacement for GAT/VGAE?

## Success Criteria

The refactor is successful when:

- The default CAN training path consumes `TemporalData`.
- A user can train without choosing node/edge packing budgets.
- Train/val/test splits are chronological and auditable.
- Unseen CAN IDs are handled by explicit policy.
- Evaluation reports event-level and segment-level behavior.
- Windowed graph code is no longer part of the normal development path.
