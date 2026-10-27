# GraphIDS contributor instructions

Read `docs/reference/research-code-style.md` for the coding and testing convention,
and `docs/responsibilities.md` for the active module map.

## Build and verification

- Lint: `ruff check graphids tests scripts`; format changed files with Ruff.
- On HPC, never run pytest or training on login nodes. Use a SLURM allocation
  and the project `.venv`; source `scripts/slurm/_preamble.sh` inside the job.
- CPU development suite: `python -m pytest -m 'not gpu and not slurm'`.
- Full suite: `python -m pytest`. CUDA cases need the pinned Mamba extra.
- Docs: `python scripts/lint_docs.py` and `mkdocs build --strict`.

## Active architecture

- YAML → `ExperimentConfig` → `RunConfig` → Ray/SLURM → Lightning `fit`/`test`.
- `graphids/exp/config.py` owns YAML inheritance and launch validation;
  `graphids/exp/runtime.py` constructs built-in and class-path components.
- `graphids/data/config.py` and `graphids/models/config.py` own component configs
  and validate direct construction. YAML types remain `can_event_dm` and
  `event_graph_mamba`; the representation kind is `can_event`.
- The event model owns its objectives. Configure `payload_weight` and
  `positive_weight`; external `loss_fn` overrides are retired.
- `graphids/data/recordings.py` owns reviewed recording provenance;
  `graphids/data/events.py` owns lossless caches and chronological independent streams.
- `graphids/models/event_graph_mamba.py` owns learned state, loss normalization,
  truncated backpropagation, and epoch-boundary checkpoints.
- `graphids/evaluation.py` owns packed frame scores and coverage validation.
- The Typer commands are in `graphids/cli/exp.py` and `graphids/cli/can.py`.
- Workers journal state and offline ingest payloads under `.graphids/`.
  `gx exp ingest` serializes shared MLflow writes after jobs complete.
- Retired snapshot experiments are reproduced from `archives/`, not live shims.

## Editing rules

- Validate at external boundaries; trust prepared objects in numerical helpers.
- Let unexpected exceptions propagate. Avoid capability probing and speculative
  defaults internally; retain concrete integration hooks and telemetry isolation.
- Keep provenance, coverage, causality, checkpoint integrity, and capacity guards.
- Prefer independent numerical/property tests over formula copies or tests of
  upstream framework guarantees. Share helpers in `tests/event_fixtures.py`.
- Use spawn multiprocessing with CUDA and preserve model training mode.
- Keep datasets, caches, run artifacts, and allocation logs outside source storage.
- Update live reference docs with architecture changes. Preserve historical
  archives and decision records.
