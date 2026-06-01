# 2026-05-31 — Snapshot-sequence split leakage plan

## Finding

`snapshot_sequence` train/val splits are graph-index disjoint, but not
underlying-window disjoint. Current construction materializes all sequence graphs
first, then randomly splits graph indices.

set_01 audit:

- train graphs: `120,902`
- val graphs: `30,225`
- exact `sequence_id` overlap: `0`
- shared underlying `snapshot_wid`s: `72,792`
- val `snapshot_wid`s shared with train: `98.46%`
- val graphs with any `snapshot_wid` in train: `30,184 / 30,225 = 99.86%`
- val graphs with all `snapshot_wid`s in train: `27,203 / 30,225 = 90.00%`

Official test splits are separate raw source dirs, so the immediate leakage is
train vs validation, not train vs test.

## Plan

Add representation-aware split planning before materialization.

Important constraint: split units must be expressed in stable data-space terms,
not just materialized graph indices. For windowed graph representations, track
both:

- dense base-window ordinal: `0..n_windows-1`
- raw row interval covered by that window: `[start_row, end_row)`

Existing `_wid` values are raw row offsets, not necessarily dense window
ordinals, so embargo widths must be explicit about which unit they use.

Default split unit by representation:

| Representation | Split unit | Embargo |
|---|---|---:|
| `snapshot` | base window ordinal + raw row interval | row/window overlap reach |
| `snapshot_sequence` | dense base-window ordinal range + raw row intervals | `(sequence_length - 1) * sequence_stride` base-window ordinals, plus row overlap reach |
| `multi_scale` | anchor ordinal plus touched scale-window raw intervals | max raw interval reach across scales |
| `temporal` | raw row/time range | history + horizon |
| `entity` | centered row/entity range | history + future reach |

For `snapshot`, embargo may be `0` only when windows are non-overlapping
(`stride >= window_size`). If `stride < window_size`, embargo must prevent
shared raw rows across train/val.

For `snapshot_sequence`, `sequence_length - 1` is not sufficient by itself when
`sequence_stride != 1` or when `_wid` is interpreted as raw row offset. Compute
the embargo in dense base-window ordinal space, then also verify raw row
interval disjointness.

Implementation sketch:

1. Add `graphids/core/data/preprocessing/splits.py`.
2. Add a `SplitPlan` dataclass with train/val/embargo units, raw row
   intervals, source/file identifiers when available, and a stable
   `split_plan_digest`.
3. Add `split_embargo_width(representation_cfg)` and make its return type
   explicit about units: dense base-window ordinals vs raw rows.
4. Materialize base window metadata before graph packing:
   `source_dir`, `source_file`, dense `window_ordinal`, raw `_wid`/start row,
   and `[start_row, end_row)`.
5. Assign base windows to train/val before final graph tables are packed.
   Exclude embargo units from both train and val.
6. Do not let train/val continue to be only random index views over the same
   `data_train.pt`. Either persist separate train/val tensors or persist a
   split mask/index artifact generated from `SplitPlan` and use it consistently
   for tensors, metadata, scalers, and dataloaders.
7. Fit `feature_scaler.pt` only on admitted train units. Validation units and
   embargo units must not contribute to scaler fitting. Include the
   `split_plan_digest` in scaler identity.
8. For `snapshot_sequence`, allow a candidate sequence into a split only when
   all underlying base windows belong to that split's allowed base-window set.
   Also assert that the sequence's raw row intervals do not intersect any
   opposite-split or embargo interval.
9. Reject or explicitly mark sequences that bridge missing base windows. Current
   sequence construction uses the post-filtered `window_ids` list, so after
   windows without edges are dropped, adjacent sequence positions may not be
   adjacent in raw time.
10. Preserve source/file boundaries. CAN rows from multiple source dirs/files
    are currently concatenated and sorted by timestamp; split planning should
    prevent windows from crossing incompatible source/file boundaries, or carry
    source/file ids into the audit and treat those boundaries as hard barriers.
11. Use blocked contiguous validation by default:

   ```text
   [ train region ][ embargo ][ val region ]
   ```

12. After planning, report label and attack-type balance for train/val. Tail
    validation may be temporally realistic but can be unrepresentative; if it
    is unstable, add blocked K-fold or multiple validation blocks.
13. Persist split policy in cache identity and metadata:
    `split_policy`, `split_unit`, `split_embargo`, `split_plan_digest`,
    `val_fraction`, `seed`, `source_boundary_policy`, and whether test vocab was
    included (`vocab_scope`).
14. Include `split_plan_digest` in on-disk cache identity, not only the
    process-level `cache_key`. Current cache root does not include
    `val_fraction`/`seed`, and `process()` can return early on stale
    `data_train.pt` plus `.complete`.
15. Add a split leakage audit helper that checks:
    - graph id overlap
    - local `sequence_id` overlap
    - underlying dense base-window ordinal overlap
    - underlying `snapshot_wid` overlap
    - raw row interval intersection
    - source_dir/source_file boundary violations
    - scaler fit units are a subset of train units
    - raw source-dir isolation for test splits
    - whether `vocab_scope="all"` used test dirs in preprocessing
16. Add regression tests for:
    - overlapping snapshot windows (`stride < window_size`)
    - non-overlapping snapshot windows (`stride >= window_size`)
    - `snapshot_sequence` with `sequence_length > 1`
    - `sequence_stride != 1`
    - missing base windows after edge filtering
    - multiple train source dirs/files with timestamp resets or overlaps
    - stale cache reuse when split policy/seed/val fraction changes
    - scaler fitting excluding val and embargo units
17. Rebuild `snapshot_sequence` caches and rerun GAT.

Expected result: validation metrics should drop from the current inflated
`val_auroc ~= 0.99999`, but become meaningful.

## Implementation status

First pass implemented:

- `graphids/core/data/preprocessing/splits.py` with blocked split planning,
  dense base-window embargo, raw row interval audit, and source-boundary
  violation audit. Source-boundary crossings are reported, not excluded, because
  current CAN loading globally timestamp-sorts multiple source files and
  excluding crossings removes all `set_01` train/val sequence graphs.
- Train/val graph datasets now derive `_indices` from the split plan instead
  of random graph indices.
- `snapshot_sequence` now uses context-to-target labels: each sample contains
  multiple snapshot windows, but `y` comes from the final/target window only.
- Feature scaler fitting now uses the split-plan train indices.
- PyG packed data now carries graph-level `graph_wid`, window row interval
  metadata, and source-boundary counts when available.
- CAN row loading now tags `source_dir` and `source_file`.
- On-disk graph cache roots and metadata include the split-plan digest.
- Cache metadata persists `split_audit`.
- Regression tests cover snapshot-sequence underlying-window separation,
  overlapping snapshot window embargo, source-boundary reporting, stale cache
  root identity, metadata invariants, and materialized window interval fields.

Still remaining:

- Add a standalone CLI/reporting helper for auditing existing built caches.
- Rebuild production `snapshot_sequence` caches and rerun GAT.
