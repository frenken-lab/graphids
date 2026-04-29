# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection via a 3-stage knowledge distillation chain:
VGAE (unsupervised reconstruction) → GAT (supervised classification) →
fusion. Large models compress into small models via KD auxiliaries for
edge deployment. Stages are trained as independent ablation presets;
multi-stage pipelines (`configs/plans/*.jsonnet`) render to a **JSONL
blueprint** via `graphids run` — the user/LLM walks the rows and
invokes `graphids submit` per node. No pipeline driver. See
`.claude/rules/single-submission-primitive.md`.

## Code Philosophy

## Key Commands

```bash
# SLURM launch via `python -m graphids submit` — one preset, real flags, no nested quotes.
# Preset owns run_dir + model/stage specifics; flags map to TLAs internally.
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --dataset set_01 --seed 42
# Fusion: --depends-on resolves teacher ckpts via MLflow (vgae → vgae_ckpt_path TLA;
# focal → gat_ckpt_path TLA). No more hand-typed paths.
python -m graphids submit configs/ablations/fusion/dqn.jsonnet \
    --dataset set_01 --seed 42 --depends-on vgae:42,focal:42 --cluster cardinal
# Idempotent re-submission: --skip-if-finished prints 0 (no submit) when MLflow
# already has a FINISHED row. Use it on every plan-blueprint row.
python -m graphids submit configs/ablations/gat_loss/focal.jsonnet \
    --dataset set_01 --seed 42 --skip-if-finished
python -m graphids submit configs/ablations/unsupervised/vgae.jsonnet --smoke --dry-run  # gpudebug 1hr

# Plan blueprint — JSONL on stdout, one row per node, each with a literal
# submit_command. Walk it manually or via jq; never auto-execute.
python -m graphids run configs/plans/ofat.jsonnet --dataset set_01 --seed 42 --cluster cardinal
python -m graphids status configs/plans/ofat.jsonnet --dataset set_01 --seed 42

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

**Training** — `python -m graphids fit|test` → `graphids/cli/training.py` (Typer). `_prepare()` renders the jsonnet, applies any `--set` overrides, builds a `ResolvedConfig.from_rendered`, wires OTel file exporters, and calls `build(resolved)`. Then `fit` calls `train(artifacts, resolved, resume_from=--ckpt-path)`; `test` calls `evaluate(artifacts, resolved)`. All three primitives (`build`/`train`/`evaluate`) plus `ResolvedConfig`, `InstantiatedRun`, and `build_run` live in the single `graphids/orchestrate.py` module — `from graphids.orchestrate import ...`. For SLURM submission, use `python -m graphids submit <preset.jsonnet> [--dataset X --seed N --scale s --cluster c]` — it builds TLAs from flags so you never type nested JSON quotes.

**Operational commands** — `graphids/cli/`. `app.py` owns the root app + shared option types (`ConfigPath`/`TlaList`/`SetList`/`CkptPath`). `--tla` and `--set` parse `key=value` via `_parse_kv_pair` (Typer `parser=` hook). Submodules register commands via `@app.command()`: `training.py`, `analysis.py`, `data.py`, `compare.py`. `graphids/__main__.py` imports submodules to register commands.

| Command                                                                     | Purpose                                   |
| --------------------------------------------------------------------------- | ----------------------------------------- |
| `python -m graphids fit` / `test`                                           | Train or evaluate one preset              |
| `python -m graphids analyze`                                                | Analysis artifacts from checkpoints       |
| `python -m graphids rebuild-caches`                                         | Rebuild preprocessed graph caches         |
| `python -m graphids extract-fusion-states`                                  | Extract VGAE+GAT latent states for fusion |
| `python -m graphids compare {leaderboard\|ties\|effect-size\|expected-max}` | Cross-variant MLflow comparison           |

**Config resolution** — Single path: `render(config_path, tla=..., set_overrides=...)` (`config/jsonnet.py`) → `ResolvedConfig.from_rendered(rendered, stage_name=<basename>)` (validates + pulls `run_dir` / `ckpt_file` from `trainer.default_root_dir`). `--set a.b.c=v` flags expand to a nested dict via `cli/app.py:dotted_to_nested` and apply via `std.mergePatch(rendered, std.extVar('overrides'))` at each ablation preset's apex (one mechanism, no Python in-place mutator). Every preset computes its own `run_dir` via `std.native('paths.run_dir')(dataset, group, variant, seed)` — `render()` registers `graphids.config.paths` functions as `native_callbacks`, so jsonnet and `slurm/dag.py` share one source of truth. `run_root` flows in via `std.extVar('run_root')` from `GRAPHIDS_RUN_ROOT` (per-user, distinct from the shared `GRAPHIDS_LAKE_ROOT` which holds mlflow.db / cache / mlartifacts). See `docs/reference/config-architecture.md`.

**SLURM submission** — one Typer command, `python -m graphids submit`, backed by `submitit.AutoExecutor` (no subprocess, no sbatch-stdout parsing). Two usage patterns: `python -m graphids submit <preset.jsonnet> [--dataset X --seed N ...]` for training (implicit `--mode gpu`, fit command), or `python -m graphids submit --mode {gpu\|cpu} --command "..." [--mem-gb N --timeout-min M]` for ops. Implementation: `graphids.slurm.submit.submit()` is the pure-Python entrypoint; the CLI wrapper lives at `graphids/cli/submit.py`; `graphids/slurm/dag.py` calls `submit()` directly. Profile JSON (`configs/resources/submit_profiles.json`) stores raw submitit kwargs keyed `[mode][cluster][length]` — no Python-side text parsing. `slurm_setup` sources `scripts/slurm/_preamble.sh` inside the sbatch shell. Optional `--time-from-history` consults MLflow for tighter walltime via `graphids.slurm.sizing`. See `rules/slurm-hpc.md`.

Fusion uses a single `configs/stages/fusion.jsonnet` that dispatches on the `fusion_method` TLA over the 4 method libsonnets in `configs/models/fusion/methods/`.

## Rules (auto-loaded from `.claude/rules/`)
