# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name>
# Stages: autoencoder, curriculum, normal, fusion, evaluation, temporal
# Models: vgae, gat, dqn | Scales: large, small | Auxiliaries: none, kd_standard

# Examples
python -m graphids.pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa
python -m graphids.pipeline.cli curriculum --model gat --scale small --auxiliaries kd_standard --teacher-path <path> --dataset hcrl_sa
python -m graphids.pipeline.cli fusion --model dqn --scale large --dataset hcrl_ch
python -m graphids.pipeline.cli temporal --model gat --scale large --dataset hcrl_sa -O temporal.enabled true
python -m graphids.pipeline.cli autoencoder --model vgae --scale large -O training.lr 0.001 -O vgae.latent_dim 16

# Full pipeline via Ray + SLURM
python -m graphids.pipeline.cli flow --dataset hcrl_sa
sbatch scripts/slurm/ray_slurm.sh flow --dataset hcrl_sa
python -m graphids.pipeline.cli flow --dataset hcrl_sa --local  # No SLURM

# Export + analytics
python -m graphids.pipeline.export                      # All exports → reports/data/ (~2s, login node OK)
python -m graphids.pipeline.export --reports            # Also copy datalake Parquet to reports/data/
python -m graphids.pipeline.build_analytics             # DuckDB rebuild (sub-second, views over Parquet)

# Tests — ALWAYS submit to SLURM
bash scripts/slurm/run_tests_slurm.sh
bash scripts/slurm/run_tests_slurm.sh -k "test_full_pipeline"

# Reports (Quarto) — site auto-deploys via GitHub Actions on push to main
quarto preview reports/                     # Dev server
quarto render reports/                      # Build → reports/_site/
quarto render reports/paper/ --to typst     # PDF output via Typst
```

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Ray pipeline to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, W&B |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Rules (auto-loaded from `.claude/rules/`)

9 modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` — full architecture, models, memory optimization
- `.claude/system/STATE.md` — current session state (updated each session)
