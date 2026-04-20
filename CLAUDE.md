# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection via a 3-stage knowledge distillation chain:
VGAE (unsupervised reconstruction) → GAT (supervised classification) →
fusion. Large models compress into small models via KD auxiliaries for
edge deployment. Stages are trained as independent ablation presets;
cross-stage chaining is a bash loop with SLURM `afterok` deps, not an
in-process driver.

## Code Philosophy

Every function, file, and abstraction must earn its place. Before writing code, answer: does a dependency already do this? Can this be inlined? Does this file need to exist or can it be 10 lines somewhere else? If you can't justify it in one sentence, delete it. When a plan says simplify — that means less code, not different code.

## Key Commands

```bash
# SLURM launch via scripts/run — one preset, real flags, no nested quotes.
# Preset owns run_dir + model/stage specifics; flags map to TLAs internally.
scripts/run configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
scripts/run configs/ablations/fusion/dqn.jsonnet \
    --dataset set_01 --seed 42 \
    --vgae-ckpt /path/checkpoints/best_model.ckpt --gat-ckpt /path/checkpoints/best_model.ckpt \
    --cluster cardinal
scripts/run configs/ablations/unsupervised/vgae.jsonnet --smoke --dry-run  # gpudebug 1hr

# Direct CLI (login-node smoke / non-SLURM).
python -m graphids fit --config configs/stages/autoencoder.jsonnet
python -m graphids fit --tla 'scale="large"' --config configs/stages/supervised.jsonnet

# Evaluation
python -m graphids test --config configs/stages/autoencoder.jsonnet --ckpt-path checkpoints/best_model.ckpt

# Analysis (auto-dispatches by ckpt class_path → model_type)
python -m graphids analyze --ckpt-path path/to/checkpoints/best_model.ckpt --dataset hcrl_sa
# Fusion models need upstream ckpts:
python -m graphids analyze --ckpt-path fusion/checkpoints/best_model.ckpt --dataset hcrl_sa \
    --vgae-ckpt vgae/checkpoints/best_model.ckpt --gat-ckpt gat/checkpoints/best_model.ckpt
```

## CLI Architecture

**Training** — `python -m graphids fit|test` → `graphids/cli/training.py` (Typer). `_prepare()` renders the jsonnet, applies any `--set` overrides, builds a `ResolvedConfig.from_rendered`, wires OTel file exporters, and calls `build(resolved)`. Then `fit` calls `train(artifacts, resolved, resume_from=--ckpt-path)`; `test` calls `evaluate(artifacts, resolved)`. For SLURM submission, use `scripts/run <preset.jsonnet> [--dataset X --seed N --scale s --cluster c]` — it builds TLAs from flags so you never type nested JSON quotes.

**Operational commands** — `graphids/cli/`. `app.py` owns the root app + shared option types (`ConfigPath`/`TlaList`/`SetList`/`CkptPath`). `--tla` and `--set` parse `key=value` via `_parse_kv_pair` (Typer `parser=` hook). Submodules register commands via `@app.command()`: `training.py`, `analysis.py`, `data.py`, `compare.py`. `graphids/__main__.py` imports submodules to register commands.

| Command | Purpose |
|---------|---------|
| `python -m graphids fit` / `test` | Train or evaluate one preset |
| `python -m graphids analyze` | Analysis artifacts from checkpoints |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids extract-fusion-states` | Extract VGAE+GAT latent states for fusion |
| `python -m graphids compare {leaderboard\|ties\|effect-size\|expected-max}` | Cross-variant MLflow comparison |
| `python -m graphids mlflow-start-parent` | Open a parent MLflow run for an ablation axis |

**Config resolution** — Single path: `render(config_path, tla=...)` (`config/jsonnet.py`) → `apply_overrides(rendered, --set ...)` (`cli/app.py`) → `ResolvedConfig.from_rendered(rendered, stage_name=<basename>)` (validates + pulls `run_dir` / `ckpt_file` from `trainer.default_root_dir`). Every preset under `configs/ablations/` computes its own `run_dir` from `(lake_root, dataset, seed)` via `_paths.libsonnet`, so there is no Python planner / identity-hash layer. See `docs/reference/config-architecture.md`.

**SLURM submission** — one script, `scripts/run`. Two usage patterns: `scripts/run <preset.jsonnet> [--dataset X --seed N ...]` for training (implicit `--mode gpu`, fit command), or `scripts/run --mode {gpu\|cpu} --command "..." [--mem M --time T]` for ops. Only two profile entries in `configs/resources/submit_profiles.json`: `gpu` and `cpu`, each with per-cluster partitions and per-length wall defaults. Per-job mem/time/command are flags, never JSON. Optional `--time-from-history` consults MLflow for tighter walltime via `graphids.slurm.sizing`. See `rules/slurm-hpc.md`.

Fusion uses a single `configs/stages/fusion.jsonnet` that dispatches on the `fusion_method` TLA over the 4 method libsonnets in `configs/fusion/methods/`.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

GitNexus CLI is available for code intelligence (`npx gitnexus query|context|impact|cypher` against the indexed graph at `.gitnexus/`). Optional — reach for it when grep/Read aren't enough. Not required before edits. Do not run `npx gitnexus analyze` — it auto-injects a block of "MUST" rules into this file.
