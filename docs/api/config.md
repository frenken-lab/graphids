# Config

The current typed config surface is split between:

- `graphids.exp.config` for run and
  experiment configs plus typed stage payloads (`FitRunPayload`,
  `CacheRunPayload`, `ExtractRunPayload`, `AnalyzeRunPayload`)
- `graphids.primitives` for data,
  model, loss, scaler, representation, and discovery primitives

The older plan-chassis documentation is kept as historical reference in
[`docs/reference/orchestration.md`](../reference/orchestration.md).

## `temporal_hybrid`

The `temporal_hybrid` model primitive exposes the newer architecture as a
modular config option in the same stack as the temporal classifier, GAT, RNN,
and VGAE baselines.

Supported top-level knobs:

- `objective`: `supervised`, `anomaly`, or `joint`
- `memory`: `type: tgn`, `enabled`, and `reset_on_stream_end`
- `backbone`: `type: none | gru | ssm_lite | mamba`, `layers`, and `dropout`
- `heads`: `classification`, `next_id`, `iat`, and `payload_delta`
- `anomaly`: `mode: regression | nll` plus optional log-scale clamps
- `rhythm`: optional causal per-ID IAT summary branch
- `motif`: optional recent destination-ID/IAT motif branch
- `loss_weights` and `anomaly_score_weights`

Head defaults follow the objective. Supervised runs enable the classification
head, anomaly runs enable the self-supervised heads, and joint runs enable both.
Invalid combinations fail during config/model construction: anomaly runs cannot
receive a classifier loss, supervised runs require classification, and anomaly
or joint runs require at least one anomaly head.

`memory.time_encoding_dim` enables CAN-TGN-style elapsed-time encodings for the
source and destination ID memories. `memory.use_source` and
`memory.use_destination` support memory ablations. `anomaly.mode: nll` changes
the IAT and payload-delta anomaly terms from SmoothL1 errors to Gaussian
negative log-likelihoods; next-ID remains categorical NLL in both modes.

Canonical smoke examples live in `configs/experiments/temporal_hybrid_*_smoke.yml`.
Final benchmark-matrix configs are intentionally separate from the core smoke
integration so the architecture can be validated before broader scheduling.
