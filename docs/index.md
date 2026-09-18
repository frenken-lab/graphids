# GraphIDS

CAN bus intrusion detection over temporal CAN event streams. The live training
path materializes raw CAN rows as PyG `TemporalData`, then trains temporal
event models through the Ray-backed experiment runner.

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
- Runtime docs (this site): built from
  [`docs/`](https://github.com/frenken-lab/graphids/tree/main/docs).
