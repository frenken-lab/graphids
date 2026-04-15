# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → fusion.
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Code Philosophy

Every function, file, and abstraction must earn its place. Before writing code, answer: does a dependency already do this? Can this be inlined? Does this file need to exist or can it be 10 lines somewhere else? If you can't justify it in one sentence, delete it. When a plan says simplify — that means less code, not different code.

## Key Commands

```bash
# Preferred: SLURM launch via scripts/run — one preset, real flags, no
# nested quotes. Preset owns run specifics; flags map to TLAs internally.
scripts/run configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
scripts/run configs/ablations/fusion/dqn.jsonnet \
    --dataset set_01 --seed 42 \
    --vgae-ckpt /path/best.ckpt --gat-ckpt /path/best.ckpt \
    --cluster cardinal
scripts/run configs/ablations/unsupervised/vgae.jsonnet --smoke --dry-run  # gpudebug 1hr

# Direct CLI (login-node smoke / non-SLURM). Stages default for zero-arg.
python -m graphids fit --config configs/stages/autoencoder.jsonnet
python -m graphids fit --tla 'scale="large"' --config configs/stages/supervised.jsonnet

# Evaluation
python -m graphids test --config configs/stages/autoencoder.jsonnet --ckpt_path best.ckpt

# Analysis artifacts (auto-dispatches by ckpt class_path → model_type)
python -m graphids analyze --ckpt-path path/to/best.ckpt --dataset hcrl_sa
# Fusion models need upstream ckpts:
python -m graphids analyze --ckpt-path fusion.ckpt --dataset hcrl_sa \
    --vgae-ckpt vgae.ckpt --gat-ckpt gat.ckpt
```

## CLI Architecture

Three entry points, zero overlap:

**Training** — `python -m graphids fit|test` → `graphids/cli/training.py` (Typer). Renders the jsonnet stage with any `--tla` flags, gates through `validate_config`, and calls `graphids.orchestrate.instantiate.instantiate(rendered) → InstantiatedRun` which handles class_path import, signature-filtered link_arguments, forced callbacks (ModelCheckpoint/EarlyStopping/DeviceStatsMonitor/ResourceProfileCallback/RunRecordCallback), logger wiring, and wandb config forwarding. Callbacks live in `graphids/core/monitoring/callbacks.py`. For SLURM submission, prefer `scripts/run <preset.jsonnet> [--dataset X --seed N --scale s --cluster c]` — it builds TLAs from flags so you never type nested JSON quotes.

**Operational commands** — Typer CLI in `graphids/cli/`. `app.py` defines the root app with shared option types (`ConfigPath`/`TlaList`/`SetList`/`CkptPath`) — `--tla` and `--set` run their `key=value` payload through `_parse_kv_pair` via Typer's `parser=` hook, and `apply_overrides` consumes the pre-parsed list-of-pairs directly. Submodules register commands via `@app.command()` decorators: `training.py`, `analysis.py`, `data.py`, `pipeline.py`. `graphids/__main__.py` imports these submodules to register all commands.

| Command | Purpose |
|---------|---------|
| `python -m graphids pipeline-run` | Run the full 3-stage chain in-process (one SLURM allocation) |
| `python -m graphids analyze` | Analysis artifacts from checkpoints (loss-landscape folded in via TLA) |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids extract-fusion-states` | Extract VGAE+GAT latent states for fusion training |

**Pipeline driver** — `graphids/orchestrate/run.py::run_pipeline(config)` loops `ResolvedConfig.resolve → build → train → evaluate` over each stage in the same Python process, with per-stage retries. Resume skip-check is authoritative on `best_model.ckpt` existence — the prior `.complete` marker (a Dagster-era workaround) was removed once the pipeline gained direct checkpoint awareness. Analysis is decoupled: run `python -m graphids analyze --ckpt-path <p>` after training. No actor framework; runs inside the SLURM allocation created by `scripts/slurm/submit.sh pipeline-run`.

**Config resolution** — Two entry points both produce a `ResolvedConfig` (frozen dataclass in `orchestrate/config.py` with fields `rendered / validated / stage_name / run_dir / ckpt_file`). **Pipeline path:** `orchestrate/resolve.py::resolve_config(cfg, lake_root, user, dataset, seed, upstream_ckpts)` builds a `PathContext` (`config/topology.py`), packs TLAs via `StageConfig.to_tla_dict(...)` — the single mapping site from schema fields to jsonnet TLA names, owned by `StageConfig` in `orchestrate/config.py` — renders the stage jsonnet via `render(...)` from `config/jsonnet.py`, and gates the output through `validate_config(...)` from `config/schemas.py`. **CLI path:** `ResolvedConfig.from_rendered(rendered, stage_name=...)`, called from `cli/training.py::_prepare` after `render(config_path, tla=...)` has already produced the rendered dict — it only validates + pulls `run_dir` / `ckpt_file` from `trainer.default_root_dir`, skipping the `PathContext` step. Adding a new TLA means editing `StageConfig.to_tla_dict` + the relevant jsonnet stage signature. See `docs/reference/config-architecture.md`.

**SLURM submission** — all jobs via `scripts/slurm/submit.sh <profile> [args]`. The preamble hard-fails if the `jsonnet` binary is missing (see ADR 0010 in `docs/decisions/README.md`). Resource profile is the single `configs/resources/submit_profiles.json` — static profiles have fixed time/mem; profiles with a `scaling` block auto-size from `cache_metadata.json.aggregate.num_raw_samples`; composed profiles (`pipeline-run`) sum per-stage time and max per-stage cpus/mem from `stage_profiles`. See `rules/slurm-hpc.md`.

Fusion uses a single `configs/stages/fusion.jsonnet` that dispatches on the `fusion_method` TLA over the 4 method libsonnets in `configs/fusion/methods/`.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

GitNexus CLI is available for code intelligence (`npx gitnexus query|context|impact|cypher` against the indexed graph at `.gitnexus/`). Optional — reach for it when grep/Read aren't enough. Not required before edits. Do not run `npx gitnexus analyze` — it auto-injects a block of "MUST" rules into this file.
