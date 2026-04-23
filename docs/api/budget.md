# Data: Budget

VRAM budget probe — sizes ``max_nodes`` and ``max_edges`` for
[``NodeBudgetBatchSampler``](sampler.md) before DataLoader
construction. Two-point linear fit of peak VRAM vs. batch size
isolates the scaling slope (``bpn_node``) from fixed overhead (cuDNN
workspaces, optimizer state, KD teacher). The slope-only estimate is
safer at high VRAM than the single-point probe it replaced, which
charged small batches with fixed cost and capped packs at ~20% of
actual VRAM.

GPS models use a quadratic probe (``peak = α·V² + β·V + γ``) to
capture attention's ``O(V²)`` blowup.

See ``.claude/rules/critical-constraints.md`` for the two-point-probe
invariant and the ``GRAPHIDS_BUDGET_SAFETY_MARGIN=0.95`` rationale.

## `graphids.core.data.budget`

::: graphids.core.data.budget
