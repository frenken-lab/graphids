# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → fusion.
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Code Philosophy

Every function, file, and abstraction must earn its place. Before writing code, answer: does a dependency already do this? Can this be inlined? Does this file need to exist or can it be 10 lines somewhere else? If you can't justify it in one sentence, delete it. When a plan says simplify — that means less code, not different code.

## Key Commands

```bash
# Training
python -m graphids fit --config graphids/config/stages/autoencoder.yaml
python -m graphids fit --config graphids/config/stages/normal.yaml --config graphids/config/models/gat/scales/small.yaml

# Evaluation
python -m graphids test --config graphids/config/stages/autoencoder.yaml --ckpt_path best.ckpt

# Analysis artifacts (embeddings, CKA, loss landscape)
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

## CLI Architecture

Three entry points, zero overlap:

**Training** — `python -m graphids fit|test|validate|predict` → `GraphIDSCLI` (extends `LightningCLI`). `GraphIDSCLI`, `WandbSaveConfigCallback`, and `CLI_KWARGS` live in `graphids/cli.py`.

**Operational commands** — registered in `_COMMAND_MODULES` dict in `__main__.py`. Convention: module name = command name (`-` → `_`), each exports `main(argv)`. Adding a subcommand = one file + one dict entry.

| Command | Purpose |
|---------|---------|
| `python -m graphids analyze` | Analysis artifacts from checkpoints |
| `python -m graphids analyze-from-spec` | Run analyzer from canonical AnalysisSpec (dagster transport) |
| `python -m graphids landscape` | 2D loss landscape |
| `python -m graphids pipeline-status` | Aggregated dagster + SLURM phase status |
| `python -m graphids profile <job_ids>` | sacct resource profiler |
| `python -m graphids profile-training` | Profiled training run (PyTorchProfiler) |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids stage-data` | NFS → scratch → TMPDIR staging |
| `python -m graphids submit-profile <job>` | Print SLURM resource profile for submit.sh |
| `python -m graphids test-from-spec` | Run test (evaluation) from canonical TrainingSpec (dagster transport) |
| `python -m graphids test-preprocessing` | Validate preprocessing pipeline |
| `python -m graphids train-from-spec` | Run training from canonical TrainingSpec (dagster transport) |

**Dagster** — own entry point, never called through `python -m graphids`:

| Command | Purpose |
|---------|---------|
| `dg launch --assets ...` | Materialize assets |
| `dg list defs` | List all assets |
| `python -m graphids.orchestrate validate` | Validate recipe config chains |

**Config resolution** — `ConfigResolver` in `orchestrate/resolve.py` is the exclusive merge path for pipeline runs. It merges trainer/resource/KD overrides, validates cross-field constraints (including YAML-aware checks via naive deep merge), and emits an audit trail. `assets.py` calls `resolver.resolve()` → `ResolvedConfig` (TrainingSpec + ResourceSpec + paths). See `issues/config-system-overhaul.md`.

**SLURM submission** — all jobs via `scripts/submit.sh <profile> [args]`. Resource profiles read from `config/resources/` (per-model profile YAMLs + `clusters.yaml` + `submit_profiles.yaml`). See `rules/slurm-hpc.md`.

Fusion uses a single `stages/fusion.yaml` + per-method overlays in `config/fusion/methods/{method}.yaml`. Config resolution in `component.py` composes the stage YAML with the method overlay from the recipe.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

> GitNexus code intelligence: See `.claude/rules/gitnexus.md`
