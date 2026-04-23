# Data: Sampler

Dual-budget bin-packing sampler — closes a batch when adding a graph
would exceed **either** the node budget or the edge budget. Single-axis
node-only budgets allowed edge-heavy batches to OOM; see
``.claude/rules/critical-constraints.md``.

Two paths:

- ``NodeBudgetBatchSampler`` — live sampler, bucket-shuffled, fresh
  each epoch. Used when ``shuffle=True``.
- ``pack_offline`` — first-fit-decreasing packing used by the prebatch
  path at setup. ~10-20% tighter than sequential; no epoch-to-epoch
  randomness to preserve.

## `graphids.core.data.sampler`

::: graphids.core.data.sampler
