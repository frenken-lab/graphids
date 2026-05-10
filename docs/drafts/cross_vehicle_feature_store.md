# Cross-Vehicle Canonical Feature Store Draft

This note captures the next data-layer step for CAN/CPS work:

1. Decode vehicle-specific DBCs locally.
2. Map decoded signal names into a shared canonical registry.
3. Persist the normalized output as a long feature table keyed by:
   - `vehicle_id`
   - `canonical_id`
   - `timestamp`
4. Build snapshot, snapshot-sequence, and multi-scale views from that shared substrate.
5. Keep batching / VRAM budgeting separate, because storage and batch composition are different problems.

## Why this helps

- Multiple vehicles can share one semantic ontology even when raw arbitration IDs differ.
- Feature reuse becomes explicit instead of being rebuilt window-by-window.
- PyG `FeatureStore` / `GraphStore` primitives become a better fit once the semantic layer is shared.
- The current budgeter can stay in place because it still answers the batch-composition question.

## Current primitive in code

- `graphids/graphids/core/data/preprocessing/canonical.py`
  - `CanonicalEntitySpec`
  - `CanonicalRegistry`
  - `CanonicalFeatureFrameSpec`
  - `build_canonical_feature_frame(...)`

## Open design questions

- What is the canonical entity unit for each vehicle family?
  - signal
  - message family
  - ECU / subsystem
  - learned latent group
- Which aliases are vehicle-specific versus globally shared?
- Should the long canonical frame be the source of truth for feature storage,
  or should it be an intermediate materialization before a `FeatureStore` wrapper?

