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

**Training** — `python -m graphids fit|test|validate|predict` → `graphids/cli/_training.py` (Typer). Renders the jsonnet stage with any `--tla` flags, gates through `validate_config`, and calls `graphids.orchestrate.instantiate.instantiate(rendered) → InstantiatedRun` which handles class_path import, signature-filtered link_arguments, forced callbacks (ModelCheckpoint/EarlyStopping/DeviceStatsMonitor/ResourceProfileCallback/RunRecordCallback), logger wiring, and wandb config forwarding. Callbacks live in `graphids/core/monitoring/callbacks.py`.

**Operational commands** — Typer CLI in `graphids/cli/`. `app.py` defines the root app with shared option types (`ConfigPath`/`TlaList`/`SetList`/`CkptPath`) — `--tla` and `--set` run their `key=value` payload through `_parse_kv_pair` via Typer's `parser=` hook, and `apply_overrides` consumes the pre-parsed list-of-pairs directly. Submodules register commands via `@app.command()` decorators: `_training.py`, `_analysis.py`, `_data.py`, `_pipeline.py`, `_slurm.py`. `graphids/__main__.py` imports these submodules to register all commands.

| Command | Purpose |
|---------|---------|
| `python -m graphids pipeline-run` | Run the full 3-stage chain in-process (one SLURM allocation) |
| `python -m graphids analyze` | Analysis artifacts from checkpoints (loss-landscape folded in via TLA) |
| `python -m graphids probe-budget` | Hardware cost model measurement across (model × scale × conv × dataset) |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids extract-fusion-states` | Extract VGAE+GAT latent states for fusion training |

**Pipeline driver** — `graphids/orchestrate/run.py::run_pipeline(config)` loops `ResolvedConfig.resolve → build → train → evaluate → run_single_analysis` over each stage in the same Python process, with per-stage retries. No actor framework; runs inside the SLURM allocation created by `scripts/slurm/submit.sh pipeline-run`.

**Config resolution** — Two entry points both produce a `ResolvedConfig` (frozen dataclass in `orchestrate/config.py` with fields `rendered / validated / stage_name / run_dir / ckpt_file`). **Pipeline path:** `orchestrate/resolve.py::resolve_config(cfg, lake_root, user, dataset, seed, upstream_ckpts)` builds a `PathContext` (`config/topology.py`), packs TLAs via `StageConfig.to_tla_dict(...)` — the single mapping site from schema fields to jsonnet TLA names, owned by `StageConfig` in `orchestrate/config.py` — renders the stage jsonnet via `render(...)` from `config/jsonnet.py`, and gates the output through `validate_config(...)` from `config/schemas.py`. **CLI path:** `ResolvedConfig.from_rendered(rendered, stage_name=...)`, called from `cli/_training.py::_prepare` after `render_config(config_path, tla=...)` has already produced the rendered dict — it only validates + pulls `run_dir` / `ckpt_file` from `trainer.default_root_dir`, skipping the `PathContext` step. Adding a new TLA means editing `StageConfig.to_tla_dict` + the relevant jsonnet stage signature. See `docs/reference/config-architecture.md`.

**SLURM submission** — all jobs via `scripts/slurm/submit.sh <profile> [args]`. The preamble hard-fails if the `jsonnet` binary is missing (see ADR 0010 in `docs/decisions/README.md`). Resource profiles read from `configs/resources/` (`job_profiles.json`, `clusters.json`, `submit_profiles.yaml`). See `rules/slurm-hpc.md`.

Fusion uses a single `configs/stages/fusion.jsonnet` that dispatches on the `fusion_method` TLA over the 4 method libsonnets in `configs/fusion/methods/`.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

GitNexus CLI is available for code intelligence (`npx gitnexus query|context|impact|cypher` against the indexed graph at `.gitnexus/`). Optional — reach for it when grep/Read aren't enough. Not required before edits. Do not run `npx gitnexus analyze` — it auto-injects a block of "MUST" rules into this file.
