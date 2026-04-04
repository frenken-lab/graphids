# GraphIDS: CAN Bus Intrusion Detection via Knowledge Distillation

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
| `python -m graphids analyze landscape` | 2D loss landscape (folded into analyze) |
| `python -m graphids pipeline-status` | Aggregated status (DuckDB catalog if available, else dagster + SLURM) |
| `python -m graphids pipeline-status --log [FILTER]` | Orchestrator event log (all/failures/retries/completions/submissions/polls) |
| `python -m graphids pipeline-status --log -f` | Follow orchestrator log (like tail -f) |
| `python -m graphids job-stats <job_ids>` | sacct resource profiler |
| `python -m graphids profile` | Profiled training run (PyTorchProfiler) |
| `python -m graphids probe-budget` | Hardware cost model measurement (multi-point, writes CSV to lake) |
| `python -m graphids.plots.budget --csv <path>` | Budget cost-model plots (Altair, polars) |
| `python -m graphids rebuild-caches` | Rebuild preprocessed graph caches |
| `python -m graphids stage-data` | NFS → scratch → TMPDIR staging |
| `python -m graphids submit-profile <job>` | Print SLURM resource profile for submit.sh |
| `python -m graphids test-from-spec` | Run test (evaluation) from canonical TrainingSpec (dagster transport) |
| `python -m graphids test-preprocessing` | Validate preprocessing pipeline |
| `python -m graphids train-from-spec` | Run training from canonical TrainingSpec (dagster transport) |
| `python -m graphids rebuild-catalog` | Rebuild DuckDB catalog from run_record.json sidecars |
| `python -m graphids _finalize-record` | (internal) Update sidecar with phases + wall_time after test+analyze |

**Dagster** — own entry point, never called through `python -m graphids`:

| Command | Purpose |
|---------|---------|
| `dg launch --assets ...` | Materialize assets |
| `dg list defs` | List all assets |
| `python -m graphids.orchestrate validate` | Validate recipe config chains |

**Config resolution** — `ConfigResolver` in `orchestrate/resolve.py` is the exclusive merge path for pipeline runs. It merges trainer/resource/KD overrides, validates cross-field constraints (including YAML-aware checks via naive deep merge), and emits an audit trail. `assets.py` calls `resolver.resolve()` → `ResolvedConfig` (TrainingSpec + ResourceSpec + paths). See frenken-lab/graphids#19.

**SLURM submission** — all jobs via `scripts/submit.sh <profile> [args]`. Resource profiles read from `config/resources/` (per-model profile YAMLs + `clusters.yaml` + `submit_profiles.yaml`). See `rules/slurm-hpc.md`.

Fusion uses a single `stages/fusion.yaml` + per-method overlays in `config/fusion/methods/{method}.yaml`. Config resolution in `component.py` composes the stage YAML with the method overlay from the recipe.

## Session Start

Always read `PLAN.md` before starting work. Update it after completing any task.

## Rules (auto-loaded from `.claude/rules/`)

modular rule files covering architecture, config, constraints, code style, SLURM, experiment tracking, PyTorch compat, shell environment, and project structure. See `.claude/rules/` directly.

> Cross-repo propagation: See `~/.claude/rules/cross-repo-propagation.md`
> Environment variables: See `~/.claude/rules/secrets-and-env-vars.md`

> GitNexus code intelligence: See `.claude/rules/gitnexus.md`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **KD-GAT** (2486 symbols, 4607 relationships, 152 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
