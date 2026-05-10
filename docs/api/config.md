# Config

The current typed config surface is split between:

- [`graphids.exp.config`](../../graphids/exp/config.py) for run and
  experiment configs plus typed stage payloads (`FitRunPayload`,
  `ExtractRunPayload`, `AnalyzeRunPayload`)
- [`graphids.primitives`](../../graphids/primitives.py) for data,
  model, loss, scaler, representation, and discovery primitives

The older plan-chassis documentation is kept as historical reference in
[`docs/reference/orchestration.md`](../reference/orchestration.md).
