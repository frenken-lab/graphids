# KD-GAT: CAN Bus Intrusion Detection via Knowledge Distillation

CAN bus intrusion detection using a 3-stage knowledge distillation pipeline:
VGAE (unsupervised reconstruction) → GAT (supervised classification) → DQN (RL fusion).
Large models are compressed into small models via KD auxiliaries for edge deployment.

## Key Commands

```bash
# Run a single stage
python -m graphids.pipeline.cli <stage> --model <type> --scale <size> --dataset <name> [--seeds <csv|count>]
# Stages: autoencoder, curriculum, normal, fusion, evaluation, temporal
# Models: vgae, gat, dqn | Scales: large, small | Auxiliaries: none, kd_standard
# Seeds: comma-separated (42,123,456) or integer count (5 = first 5 defaults)

# Examples
python -m graphids.pipeline.cli autoencoder --model vgae --scale large --dataset hcrl_sa
python -m graphids.pipeline.cli curriculum --model gat --scale small --auxiliaries kd_standard --teacher-path <path> --dataset hcrl_sa
python -m graphids.pipeline.cli fusion --model dqn --scale large --dataset hcrl_ch
python -m graphids.pipeline.cli temporal --model gat --scale large --dataset hcrl_sa -O temporal.enabled true
python -m graphids.pipeline.cli autoencoder --model vgae --scale large -O training.lr 0.001 -O vgae.latent_dim 16

# Full pipeline via Ray + SLURM
python -m graphids.pipeline.cli flow --dataset hcrl_sa
sbatch scripts/slurm/ray_slurm.sbatch flow --dataset hcrl_sa
python -m graphids.pipeline.cli flow --dataset hcrl_sa --local  # No SLURM

# Multi-seed pipeline (for statistical significance)
python -m graphids.pipeline.cli flow --dataset hcrl_sa --seeds 42,123,456
sbatch --account=$KD_GAT_SLURM_ACCOUNT scripts/slurm/ray_slurm.sbatch flow --dataset hcrl_sa --seeds 42,123
python -m graphids.pipeline.cli autoencoder --model vgae --scale large --seeds 42,123,456 --dataset hcrl_sa

# Analytics
python scripts/data/push_experiments_to_hf.py           # MLflow → HF Dataset for dashboard
# Tests — ALWAYS submit to SLURM
bash scripts/slurm/run_tests_slurm.sh
bash scripts/slurm/run_tests_slurm.sh -k "test_full_pipeline"
```

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Skills

| Skill | Usage | Description |
|-------|-------|-------------|
| `/run-pipeline` | `/run-pipeline hcrl_sa large` | Submit Ray pipeline to SLURM |
| `/check-status` | `/check-status hcrl_sa` | Check SLURM queue, checkpoints, MLflow |
| `/run-tests` | `/run-tests` or `/run-tests test_config` | Run pytest suite |
| `/sync-state` | `/sync-state` | Update STATE.md from current outputs |

## Rules (auto-loaded from `.claude/rules/`)

9 modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

## Detailed Documentation

- `.claude/system/PROJECT_OVERVIEW.md` — full architecture, models, memory optimization (updated 2026-03-07)
- `.claude/system/STATE.md` — current session state (updated 2026-03-07)

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **KD-GAT** (3973 symbols, 5935 relationships, 160 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## When Debugging

1. `gitnexus_query({query: "<error or symptom>"})` — find execution flows related to the issue
2. `gitnexus_context({name: "<suspect function>"})` — see all callers, callees, and process participation
3. `READ gitnexus://repo/KD-GAT/process/{processName}` — trace the full execution flow step by step
4. For regressions: `gitnexus_detect_changes({scope: "compare", base_ref: "main"})` — see what your branch changed

## When Refactoring

- **Renaming**: MUST use `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` first. Review the preview — graph edits are safe, text_search edits need manual review. Then run with `dry_run: false`.
- **Extracting/Splitting**: MUST run `gitnexus_context({name: "target"})` to see all incoming/outgoing refs, then `gitnexus_impact({target: "target", direction: "upstream"})` to find all external callers before moving code.
- After any refactor: run `gitnexus_detect_changes({scope: "all"})` to verify only expected files changed.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Tools Quick Reference

| Tool | When to use | Command |
|------|-------------|---------|
| `query` | Find code by concept | `gitnexus_query({query: "auth validation"})` |
| `context` | 360-degree view of one symbol | `gitnexus_context({name: "validateUser"})` |
| `impact` | Blast radius before editing | `gitnexus_impact({target: "X", direction: "upstream"})` |
| `detect_changes` | Pre-commit scope check | `gitnexus_detect_changes({scope: "staged"})` |
| `rename` | Safe multi-file rename | `gitnexus_rename({symbol_name: "old", new_name: "new", dry_run: true})` |
| `cypher` | Custom graph queries | `gitnexus_cypher({query: "MATCH ..."})` |

## Impact Risk Levels

| Depth | Meaning | Action |
|-------|---------|--------|
| d=1 | WILL BREAK — direct callers/importers | MUST update these |
| d=2 | LIKELY AFFECTED — indirect deps | Should test |
| d=3 | MAY NEED TESTING — transitive | Test if critical path |

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/KD-GAT/context` | Codebase overview, check index freshness |
| `gitnexus://repo/KD-GAT/clusters` | All functional areas |
| `gitnexus://repo/KD-GAT/processes` | All execution flows |
| `gitnexus://repo/KD-GAT/process/{name}` | Step-by-step execution trace |

## Self-Check Before Finishing

Before completing any code modification task, verify:
1. `gitnexus_impact` was run for all modified symbols
2. No HIGH/CRITICAL risk warnings were ignored
3. `gitnexus_detect_changes()` confirms changes match expected scope
4. All d=1 (WILL BREAK) dependents were updated

## CLI

- Re-index: `npx gitnexus analyze`
- Check freshness: `npx gitnexus status`
- Generate docs: `npx gitnexus wiki`

<!-- gitnexus:end -->
