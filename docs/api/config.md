# Config

The current typed config surface is split between:

- `graphids.exp.config` for run and
  experiment configs plus typed stage payloads (`FitRunPayload`,
  `CacheRunPayload`, `ExtractRunPayload`, `AnalyzeRunPayload`)
- `graphids.primitives` for data,
  model, loss, scaler, representation, and discovery primitives

The older plan-chassis documentation is kept as historical reference in
[`docs/reference/orchestration.md`](../reference/orchestration.md).
