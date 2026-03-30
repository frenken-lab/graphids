# wandb — Decisions & Reference

> Implemented: 2026-03-30 | All checklist items done

## What's wired

- **WandbLogger + CSVLogger** in `trainer.yaml` (no `save_dir` — uses `WANDB_DIR` env var)
- **WandbSaveConfigCallback** in `cli.py:9-17` — forwards full jsonargparse config (Lightning #19728 workaround)
- **Env vars** in `_preamble.sh:25-27`: `WANDB_DIR=/fs/scratch/PAS1266/wandb`, `WANDB_DISABLE_GIT=true`, `WANDB_SILENT=true`
- **Auth**: `~/.netrc` (entity: `frenken-2-the-ohio-state-university`)
- **Fallback**: `WANDB_MODE=offline` + `wandb sync` if network flakes (not wired — use manually)

## Decisions (don't re-investigate)

| Area | Verdict | Key reason |
|------|---------|------------|
| Sweeps | **SKIP** | Agent is launcher → conflicts with dagster. Use Optuna + `WeightsAndBiasesCallback` for HPO. |
| Model Registry | **SKIP** | `log_model: false`. 7-18 GB/round of 100 GB quota. Filesystem + DuckDB sufficient. |
| Data Artifacts | **SKIP** | `add_reference()` doesn't support `file://`. Existing staging protocol works. |
| dagster-wandb | **SKIP** | Training in SLURM subprocess, not dagster process. IO Manager would create empty runs. |
| OmegaConf | **INCOMPATIBLE** | `parser_mode="omegaconf"` only resolves within single YAML file; multi-file chain breaks. jsonargparse `env_prefix` handles env vars. |

## Adoption history (context for next time someone suggests removing it)

Third attempt. wandb v1 (Feb–Mar 2026) had custom wrappers (`_init_wandb`, `_finish_wandb`) — those were the complexity, not wandb itself. MLflow replaced it briefly, then both removed in simplification campaign. This time: LightningCLI handles lifecycle, zero custom code. Don't add wrappers.

## Pricing

Academic tier: 100 GB free, all Pro features. 36 runs ≈ 200-400 MB. Rate limit: 200 req/min.

## Sources

- [System metrics](https://docs.wandb.ai/models/ref/python/experiments/system-metrics) | [Env vars](https://docs.wandb.ai/guides/track/environment-variables) | [Offline mode](https://docs.wandb.ai/support/run_wandb_offline/)
- [Lightning #19728](https://github.com/Lightning-AI/pytorch-lightning/issues/19728) — WandbSaveConfigCallback workaround
- [Optuna + wandb](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html)
