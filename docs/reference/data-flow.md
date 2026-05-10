# Pipeline Data Flow

This page is a compatibility note, not the primary architecture spec.

The current canonical overview lives in:

- [`docs/reference/data-architecture.md`](./data-architecture.md)

Legacy training details that still matter:

- raw CAN rows are normalized, parsed, and cached before graph materialization
- graph materialization receives an explicit segment config derived from
  `representation_cfg` at the pipeline boundary
- the runtime loader still uses budget-aware batching for variable-size graphs

What changed:

- representation config is now the primary user-facing surface
- raw storage, materialized views, and discovery/hypothesis data are split
- snapshot, temporal, multi-scale, sequence, and entity are explicit
  representations
