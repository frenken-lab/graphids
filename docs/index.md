# GraphIDS

CAN bus intrusion detection with graph–Mamba models. New learning work uses
[message-preserving inputs](reference/graph-mamba-events.md): individual CAN
frames enter a bus-sequence Mamba encoder and learned causal graph readout.
The [snapshot implementation](reference/graph-mamba.md) is archived for reproduction.
The active Ray/Lightning/Slurm experiment path uses continuous event history and
independent recording streams on one GPU.

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
