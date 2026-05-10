# Runtime

The experiment runtime is the live launch surface. It accepts a typed
`RunConfig`, writes the manifest and event log, and dispatches the
stage-specific work directly:

- `fit` / `test` → Lightning trainer launch with config-driven data/model
  instantiation
- `extract` → feature extraction over configured checkpoints and dataset
- `analyze` → per-checkpoint artifact generation through
  `graphids.core.artifacts.analyzer.Analyzer`

The runtime module is intentionally narrow and replaces the old
row/orchestrate chassis.

## `graphids.exp.runtime`

::: graphids.exp.runtime
