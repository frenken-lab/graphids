# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → fusion.
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Code Philosophy

Every function, file, and abstraction must earn its place. Before writing code, answer: does a dependency already do this? Can this be inlined? Does this file need to exist or can it be 10 lines somewhere else? If you can't justify it in one sentence, delete it. When a plan says simplify — that means less code, not different code.

## Key Commands

```bash
# Training — jsonnet stages (Phase 1 migration 2026-04-05). Each stage
# has sensible TLA defaults so zero-arg invocation works as a smoke test.
python -m graphids fit --config configs/stages/autoencoder.jsonnet
python -m graphids fit --tla 'scale="large"' --config configs/stages/supervised.jsonnet

# Pass TLAs to stages (JSON-encoded values; unquoted bare strings also accepted)
python -m graphids fit \
    --tla 'dataset="hcrl_sa"' \
    --tla 'fusion_method="dqn"' \
    --config configs/stages/fusion.jsonnet \
    --model.init_args.lr=0.005

# Evaluation
python -m graphids test --config configs/stages/autoencoder.jsonnet --ckpt_path best.ckpt

# Analysis artifacts (single config dispatches by model_type)
python -m graphids analyze --config configs/stages/analyze.jsonnet \
    --tla 'model_type="vgae"' --analyzer.ckpt_path path/to/best.ckpt \
    --analyzer.dataset hcrl_sa
```

## CLI Architecture

Three entry points, zero overlap:

**Training** — `python -m graphids fit|test|validate|predict` → `graphids/cli/_training.py` (Typer). Renders the jsonnet stage with any `--tla` flags, gates through `validate_config`, and calls `graphids.instantiate.instantiate(rendered) → InstantiatedRun` which handles class_path import, signature-filtered link_arguments, forced callbacks (ModelCheckpoint/EarlyStopping/DeviceStatsMonitor/ResourceProfileCallback/RunRecordCallback), logger wiring, and wandb config forwarding. Callbacks live in `graphids/core/monitoring/callbacks.py`.

**Operational commands** — Typer CLI in `graphids/cli/`. `app.py` defines the root app with shared options (`parse_tla`, `apply_overrides`). Submodules register commands via `@app.command()` decorators: `_training.py`, `_analysis.py`, `_data.py`, `_orchestrate.py`, `_slurm.py`. `graphids/__main__.py` imports these submodules to register all commands.

| Command | Purpose |
|---------|---------|
| `python -m graphids analyze` | Analysis artifacts from checkpoints |
| `python -m graphids analyze landscape` | 2D loss landscape (folded into analyze) |
| `python -m graphids pipeline-status` | Aggregated status (DuckDB catalog + planner topology) |
| `python -m graphids job-stats <job_ids>` | sacct resource profiler |
| `python -m graphids profile` | Profiled training run (PyTorchProfiler) |
| `python -m graphids probe-budget` | Hardware cost model measurement (multi-point, writes CSV to lake) |
| `python -m graphids.plots.budget --csv <path>` | Budget cost-model plots (Altair, polars) |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids stage-data` | NFS → scratch → TMPDIR staging |
| `python -m graphids submit-profile <job>` | Print SLURM resource profile for submit.sh |
| `python -m graphids rebuild-catalog` | Rebuild DuckDB catalog from run_record.json sidecars |
| `python -m graphids _finalize-record` | (internal) Update sidecar with phases + wall_time after test+analyze |

**Config resolution** — `resolve_config()` in `orchestrate/resolve.py` is the exclusive merge path for pipeline runs. It packs trainer/resource/KD overrides into a typed TLA dict via `graphids.orchestrate.contracts.build_tla_dict`, renders the stage jsonnet via `graphids.config.jsonnet.render_config`, and validates cross-field constraints. Monarch actors call `resolve_config(cfg, ...)` → `ResolvedConfig` (TrainingSpec + ResourceSpec + paths). See `docs/reference/config-architecture.md`.

**SLURM submission** — all jobs via `scripts/slurm/submit.sh <profile> [args]`. The preamble hard-fails if the `jsonnet` binary is missing (see `docs/decisions/0010-jsonnet-binary.md`). Resource profiles read from `configs/resources/` (`job_profiles.json`, `clusters.json`, `submit_profiles.yaml`). See `rules/slurm-hpc.md`.

Fusion uses a single `configs/stages/fusion.jsonnet` that dispatches on the `fusion_method` TLA over the 4 method libsonnets in `configs/fusion/methods/`.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

GitNexus CLI is available for code intelligence (`npx gitnexus query|context|impact|cypher` against the indexed graph at `.gitnexus/`). Optional — reach for it when grep/Read aren't enough. Not required before edits. Do not run `npx gitnexus analyze` — it auto-injects a block of "MUST" rules into this file.
