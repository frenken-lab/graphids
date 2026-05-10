# Data Storage Layout Draft

This draft is the concrete split for the CAN/CPS data layer.

## 1. Raw storage

Source of truth: immutable CAN rows.

Recommended tables / files:

- `raw_can_events`
  - `vehicle_id`
  - `timestamp`
  - `arb_id`
  - `payload`
  - `attack`
  - `attack_type`
  - any source-specific provenance columns

Optional extra raw fields:

- `signal_hint`
- `dbc_name`
- `source_file`
- `source_subdir`

Python contract:

- [`RawEventTableSpec`](../../graphids/core/data/discovery/layout.py)

## 2. Materialized views

Training-facing representations derived from raw storage.

Recommended views:

- `snapshot`
  - one graph per window
- `snapshot_sequence`
  - ordered list of snapshot graphs
- `multi_scale`
  - multiple window sizes over the same raw source
- `temporal`
  - event stream / `TemporalData`
- `entity`
  - centered on one canonical entity or raw signal family

Recommended columns:

- `vehicle_id`
- `view_kind`
- `split`
- `timestamp` or `window_start` / `window_end`
- `canonical_id`
- feature columns
- label columns
- provenance columns

Python contract:

- [`MaterializedViewSpec`](../../graphids/core/data/discovery/layout.py)

## 3. Hypothesis store

This is the layer for “we do not know the DBC, but this raw signal behaves like RPM.”

Recommended table:

- `canonical_hypotheses`
  - `vehicle_id`
  - `raw_signal`
  - `candidate_canonical_id`
  - `confidence`
  - `status`
  - `evidence`
  - `profile_path`
  - `feature_digest`

Recommended profile table:

- `signal_profiles`
  - `vehicle_id`
  - `arb_id`
  - `signal_key`
  - message counts
  - timing stats
  - byte statistics
  - entropy stats
  - attack stats

Python contracts:

- [`SignalProfileSpec`](../../graphids/core/data/discovery/hypotheses.py)
- [`SignalHypothesisSpec`](../../graphids/core/data/discovery/hypotheses.py)
- [`DiscoveryStore`](../../graphids/core/data/discovery/hypotheses.py)

## Read order for training

1. Raw storage is the source of truth.
2. Materialized views are what training should consume first.
3. Hypotheses annotate and improve the raw-to-view mapping.

In practice:

- baseline snapshot training reads from materialized views
- temporal / sequence training may read from raw storage plus a segmenter
- hypothesis discovery writes profile and hypothesis tables alongside the cache

