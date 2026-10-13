# GraphIDS

CAN bus intrusion detection with a shared graph-Mamba model. Causal physical-time
snapshots feed per-ID temporal Mamba blocks, a same-time graph readout, and
separate classification and normal-forecasting heads. The Ray-backed experiment
runner supports pooled CAN datasets and data parallel training.

Read [Graph Mamba training and evaluation](reference/graph-mamba.md) for the
recording manifest, architecture, migration, and functional gates.

## Where to start

- **[Module responsibilities](responsibilities.md)** — one-page map of
  what every layer owns from experiment YAML through runtime execution.
- **[Config system](reference/config-architecture.md)** — how an
  experiment YAML becomes a cache build, training run, or analysis job.
- **[Data architecture](reference/data-architecture.md)** — raw rows,
  explicit representations, materialized views, and discovery/hypotheses.
- **[Decisions](decisions/README.md)** — the ADR log. Permanent
  verdicts on the tools and patterns that got adopted or rejected.
- **[API Reference](api/config.md)** — auto-generated from docstrings.

## Source

- Repo: <https://github.com/frenken-lab/graphids>
- CAN decoding and relationship research: [can-discovery](https://github.com/frenken-lab/can-discovery) (private lab repository).
- Runtime docs (this site): built from
  [`docs/`](https://github.com/frenken-lab/graphids/tree/main/docs).
