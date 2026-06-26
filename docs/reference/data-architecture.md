# Data Architecture

This is the current GraphIDS data layout after the preprocessing refactor.

## 1. Raw storage

Source of truth: immutable CAN/CPS rows.

Typical fields:

- `vehicle_id`
- `timestamp`
- `arb_id`
- `payload`
- `attack`
- `attack_type`
- provenance fields from the source

Code surface:

- `graphids/core/data/datasets/can_bus.py`
- `graphids/core/data/datasets/_base.py`

## 2. Representations

The primary public representation kinds are:

- `snapshot`
- `snapshot_sequence`
- `multi_scale`
- `temporal`
- `entity`

Representation configs live in:

- `graphids/core/data/preprocessing/representations.py`

They bridge to:

- snapshot representation configs
- snapshot-sequence representation configs
- leakage-safe split policy metadata

## 3. Materialized views

Training-facing materializations are derived from raw storage through the
selected representation.

Examples:

- snapshot graphs
- snapshot sequences

Code surface:

- `graphids/core/data/preprocessing/representations.py`
- `graphids/core/data/preprocessing/materialization.py`
- `graphids/core/data/preprocessing/pyg.py`
- `graphids/core/data/preprocessing/splits.py`

## 4. Discovery and hypotheses

This layer stores signal profiles and provisional canonical mappings.
It is where hidden-DBC cross-vehicle alignment lives.

Typical records:

- raw signal profile tables
- canonical hypotheses
- confidence
- evidence
- provenance

Code surface:

- `graphids/core/data/discovery/hypotheses.py`
- `graphids/core/data/discovery/canonical.py`
- `graphids/core/data/discovery/layout.py`

## 5. Selection rule

The primary user-facing control surface is now:

- `representation_cfg`

Window sizes and strides are resolved from the representation config at the
pipeline boundary, which derives an explicit segment config before
materialization.

## 6. Training flow

Recommended read order:

1. raw storage
2. representation selection
3. materialized views
4. hypothesis annotations

The training path should consume the materialized view that matches the
representation, while the discovery path writes the signal profile and
hypothesis tables alongside the cache.
