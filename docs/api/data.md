# Core: Data

All dataset, batching, and feature-pipeline machinery. Top-level
modules:

- **`budget`** — VRAM budget probe; sizes `max_nodes` + `max_edges`
  for `NodeBudgetBatchSampler` before DataLoader construction.
  Two-point linear fit of peak VRAM vs. batch size isolates the
  scaling slope (`bpn_node`) from fixed overhead. GPS models use a
  quadratic probe to capture attention's `O(V²)` blowup. See
  [`critical-constraints.md`](https://github.com/frenken-lab/graphids/blob/main/.claude/rules/critical-constraints.md)
  for the two-point probe invariant and `GRAPHIDS_BUDGET_SAFETY_MARGIN=0.95`.
- **`sampler`** — dual-budget bin-packing. `NodeBudgetBatchSampler`
  is the live sampler (bucket-shuffled, fresh each epoch);
  `pack_offline` is the FFD prebatch path (~10–20% tighter, no
  epoch randomness).
- **`vocab`** — shared `arb_id → index` map across train/val/all
  test subdirs so attack-injected IDs don't overflow the embedding
  table sized for train. Index 0 reserved for UNK; SHA256 over
  `(id, index)` pairs is the cache invariant.
- **`datamodule`** — `GraphDataModule` / `FusionDataModule`. Single
  `bind(*, model, device)` seam (replaces the old `_set_*` pair).
  VRAM probe is delegated to the model via
  `GraphModuleBase.compute_budget`.
- **`scaler`**, **`cache`**, **`metadata`**, **`graph_pipeline`**,
  **`fusion_states`**, **`curriculum`**, **`rebuild`** — supporting
  pipeline stages.

## `graphids.core.data`

::: graphids.core.data
    options:
      show_submodules: true
