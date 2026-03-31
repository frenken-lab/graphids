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
python -m graphids fit --config graphids/config/stages/normal.yaml --config graphids/config/overlays/small_gat.yaml

# Evaluation
python -m graphids test --config graphids/config/stages/autoencoder.yaml --ckpt_path best.ckpt

# Analysis artifacts (embeddings, CKA, loss landscape)
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

## CLI Architecture

`GraphIDSCLI`, `WandbSaveConfigCallback`, and `CLI_KWARGS` live in `graphids/cli.py` — single definition shared by
`__main__.py` (training) and `orchestrate/` (validation). `WandbSaveConfigCallback` forwards
full jsonargparse config to wandb (Lightning #19728 workaround). Two entry points:

- `python -m graphids fit|test|validate|predict` → `GraphIDSCLI` (extends `LightningCLI`, adds `link_arguments` for DRY config)
- `python -m graphids analyze` → `Analyzer` class (no Trainer — loads checkpoints, generates artifacts)
- `python -m graphids profile <job_ids>` → sacct resource profiler (RSS, CPU%, wall time). See `orchestrate/profiler.py`.

Orchestration: `python -m graphids.orchestrate [run|validate|smoke]` — dagster-based, see `plans/architecture/dagster-native-orchestration.md`

Dagster UI: `bash scripts/dev/dagster-ui.sh` (webserver + daemon on login node, port 3000). Access via SSH tunnel.

Fusion has per-method stage YAMLs: `fusion_bandit.yaml` (`BanditFusionModule`), `fusion_dqn.yaml` (`DQNFusionModule`), `fusion_mlp.yaml` (`MLPFusionModule`), `fusion_weighted_avg.yaml` (`WeightedAvgModule`). Config resolution in `component.py` picks the right YAML from `fusion_method` in the recipe.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

> GitNexus code intelligence: See `.claude/rules/gitnexus.md`
