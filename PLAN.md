# KD-GAT Session Plan

> Last updated: 2026-03-28

## Active Plan

### Ablation Run 004 — Ready to submit

Run 003 (Hydra-era, 2026-03-25) checkpoints are incompatible with post-flatten code.
Re-training as Run 004 with all 18 configs including KD (configs 10-11).

18 configs x 2 datasets (set_01, set_02) x 1 seed (42). KD configs now wired.

**Verify after Run 003 completes:**

- [ ] Each ablation config produces a unique run directory (hash suffix)
- [ ] Shared upstream stages (VGAE autoencoder) are not duplicated
- [ ] DuckDB catalog has rows with `identity_hash IS NOT NULL`
- [ ] `metrics.json` exists in evaluation run dirs
- [ ] VRAM utilization improved (target: 8-12 GB of 16 GB with batch_size=8192)
- [ ] No timeouts at 240 min wall time
- [ ] GPS conv_gps jobs complete without OOM (VRAM-aware cap ~20K nodes)
- [ ] DGI (unsup_dgi) trains and evaluates successfully
- [ ] `load_from_checkpoint()` round-trips correctly at stage boundaries

**Status tracking:** `sacct -u $USER --starttime=<submit_time>`

### IO Inconsistencies

Expanded configs now write to `{lake_root}/expanded/` (ESS on OSC). Remaining:
- Slurm logs still write to `slurm_logs/` in repo
- Test outputs still write to repo

### Configs (18 runnable)

| Claim                  | Configs | What varies                              |
| ---------------------- | ------- | ---------------------------------------- |
| Loss x Curriculum      | 6       | ce/focal/weighted_ce x curriculum/normal |
| Fusion method          | 4       | bandit/dqn/mlp/weighted_avg              |
| Conv type              | 3       | gatv2/gatv1/gps                          |
| Unsup method           | 3       | vgae/gae/dgi                             |
| Single-model baselines | 2       | vgae_only/gat_only                       |

### KD pipeline (configs 10-11)

Config 11 (large reference) trains first. Config 10 (KD student) depends on it — pass
teacher checkpoint path via `--model.init_args.auxiliaries[0].model_path=<path>` at submit time.

## In Progress

- Ops dashboard (`buckeyeguy/kd-gat-dashboard`) -- running on HF Spaces
- **Ablation Run 004** -- failed 2026-03-29 (100% failure). Postmortem: `plans/dagster-ablation-postmortem.md`.
  P0-P2 fixes applied. P2.5 collapse done (expand.py deleted, dagster reads recipe directly).
  Next: `python -m graphids.orchestrate validate`, then smoke on gpudebug, then resubmit.

### Code consolidation (deferred)

- [ ] Models consolidation (`plans/models-consolidation.md`) -- registry.py dissolution, GraphModuleBase, optimizer wiring
- [ ] Preprocessing consolidation (`plans/preprocessing-consolidation.md`) -- delete _temporal.py, DataModule convention fixes

## Recently Completed

### P2.5: Collapse expand.py into dagster_defs.py (2026-03-29)

Eliminated the two-phase expand→manifest→dagster pipeline. `dagster_defs.py` now
reads `ablation.yaml` directly, computes topology and identity hashes in-process
(no torch import at definition time), and builds multi-config SLURM commands.

Deleted: `expand.py` (420 lines), `expanded_dir()`, 64 expanded YAMLs + manifest.json.
Changed: `generate_script` accepts `config_files: list[str]` (multi-config flags),
`run_dir()` added to `config/__init__.py`, `orchestrate/__main__.py` gains
`validate`/`smoke` subcommands.

Net: 420 lines deleted (expand.py), ~80 lines added to dagster_defs.py.
SLURM command now: `python -m graphids fit --config stages/X.yaml --config overlays/Y.yaml --model.init_args.foo=bar`

### Dagster Phase C+D: config expansion + dynamic asset graph (2026-03-29)

Verified `trainer.yaml` wiring (all 4 stages get callbacks, mixed precision).
Fixed fusion identity keys: added `conv_type`, `variational` to prevent incorrect
dedup across conv types/unsup methods. Added `variational` to curriculum identity.
Added identity key metadata params to `RLFusionModule` and `GATModule`.
Wrote `ablation.yaml` (18 configs) + `expand.py` (150 lines) + rewrote `dagster_defs.py`
(175 lines) with dynamic asset factory from manifest topology.

32 unique assets (6 autoencoders, 8 curricula, 3 normals, 15 fusions) × 2 datasets
= 64 expanded YAMLs. DAG deps wired from `STAGE_DEPENDENCIES` + KD cross-pipeline.
Upstream checkpoint paths resolved at materialization time. Dry-run `RUN_SUCCESS`
for `set_01|42` (all 32 assets). Added missing resource profiles (dqn/small/large,
dgi/small). Upgraded alembic 1.6.5→1.18.4 (SQLAlchemy 2.0 compat).

### Dagster orchestrate rewrite + gpudebug spike (2026-03-28)

Replaced `graphids/orchestrate/` with dagster-based system. Deleted `submit.py` (247
lines hand-rolled Pipeline class). New files: `slurm.py` (102 lines, sbatch/sacct),
`dagster_defs.py` (140 lines, asset factory + partitions + retry). Config expansion
via jsonargparse `--print_config`. dagster-slurm rejected (requires SSH). Pipes
protocol rejected (post-hoc metrics sufficient). Added `small` scale resource profiles.
Removed 6 dead hydra packages. See `plans/orchestrate-rewrite.md`.

Gpudebug spike (job 46121143) validated full loop: dagster → sbatch → poll → COMPLETED.
Bugs found and fixed: `link_arguments` for model→data params (`conv_type`, `heads`),
`compute_node_budget` replaced with VRAM-driven `vram_node_budget` (uses
`torch.cuda.mem_get_info`), alembic upgraded for SQLAlchemy 2.0. Discovered
`trainer.yaml` is dead config (not loaded) — blocks Phase C.

### KD wiring + bug fixes (2026-03-28)

Fixed 3 KD bugs: `teacher_on_device` stale nested ref, `prepare_kd` identity hash using
student cfg for teacher path, mixed `.get()`/`getattr()` on hparams. Created 4 overlay YAMLs
(`large_vgae`, `large_gat`, `kd_vgae`, `kd_gat`). Fixed `CATALOG_PATH` missing from config
exports. Fixed stale `"pipeline"` lazy import. Fixed `orchestrate/resources.py` stale path.
Run 003 checkpoints declared incompatible (Hydra-era nested format) — re-training as Run 004.

### Artifacts `analyze` subcommand (2026-03-28)

`python -m graphids analyze --config stages/analyze_vgae.yaml --analyzer.ckpt_path ... --analyzer.dataset ...`. Same jsonargparse, YAML under `analyzer:` namespace. `Analyzer` class in `graphids/core/artifacts/analyzer.py`. Fail-loud on missing checkpoints/deps.

### Config flatten + consolidation (2026-03-28)

Replaced Hydra/OmegaConf + config dataclasses with jsonargparse + flat YAML. All 5 LightningModules take flat typed primitives. Deleted `schema.py`, `coerce_config`, `resolve()`, `defaults/` directory. See `plans/flatten-model-config.md`.

### Lightning callback extraction + LightningCLI (2026-03-27)

Replaced handrolled runner.py orchestration with Lightning callbacks + LightningCLI. `GraphIDSCLI` in `graphids/__main__.py`. Deleted entire `graphids/pipeline/` package (callbacks.py, cli.py, manifest.py, runner.py, stages/).

### Config system rewrite (2026-03-26)

Replaced Hydra/OmegaConf with jsonargparse + plain YAML. Config package: `__init__.py` (constants + topology + path helpers), `constants.yaml`, `pipeline.yaml`, `datasets.yaml`, `resources.yaml`, `trainer.yaml`, `stages/*.yaml`, `overlays/*.yaml`.

### Codebase cleanup (2026-03-25)

Replaced custom DataLoader/collation/assembly with PyG APIs, adopted Lightning built-ins.

## Blocked

- **Ablation Run 004 eval** -- blocked on training completion. After all jobs finish:
  `python -m graphids test` per run dir, then aggregate results to DuckDB catalog.
- **HPO sweep** -- blocked on ablation results + Optuna integration (Phase 2)
- **Full pipeline** -- blocked on HPO results (Phase 3)

## Open Questions

- VGAE worker memory bloat (13G vs 22G bimodal) -- same model, different nodes. PrefetchLoader may help, needs rerun to confirm.
- `--mem` over-requesting 54G when peak is 23G -- `resources.yaml` updated to 24-32G range. Validate in gpudebug spike.
- ~~dagster-slurm plugin vs custom PipesSlurmClient~~ -- **RESOLVED.** Custom `slurm.py` (99 lines). dagster-slurm requires SSH.

## Current Architecture

### CLI entry points (`graphids/__main__.py`)

```bash
# Training (GraphIDSCLI -> LightningCLI)
python -m graphids fit --config graphids/config/stages/autoencoder.yaml

# Analysis artifacts (Analyzer -- no Trainer)
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

### Config layout (`graphids/config/`)

```
__init__.py          # constants, topology, path helpers (single Python file)
constants.yaml       # static values
pipeline.yaml        # DAG topology: stages, dependencies, identity_keys
datasets.yaml        # dataset catalog (YAML anchors)
resources.yaml       # SLURM resource profiles
trainer.yaml         # default_config_files: seed, trainer
stages/              # one per stage + analyze configs
overlays/            # thin scale/ablation variants
```

### Orchestration (`graphids/orchestrate/`)

```
__init__.py          # package docstring
__main__.py          # CLI: run (dagster), validate, smoke subcommands
slurm.py             # sbatch submit, sacct poll, multi-config script gen
dagster_defs.py      # recipe→topology, asset factory, validate, smoke, Definitions
resources.py         # ResourceSpec + scale_resources (reads resources.yaml)
```

### Key Reference Documents

- `plans/dagster-native-orchestration.md` -- **active**: replace custom code with dagster-slurm + Component + IOManager
- `plans/dagster-history.md` -- archived: timeline, lessons, postmortem from dagster build
- `plans/experiment-sweep-plan.md` -- ablation claims, configs, stage sharing DAG
- `plans/tier-priority-and-implementation.md` -- priority-ordered task list
- `plans/models-consolidation.md` -- deferred: registry dissolution, shared base
- `plans/preprocessing-consolidation.md` -- deferred: delete _temporal.py, DataModule fixes
- `plans/flatten-model-config.md` -- completed: config flatten reference
- `plans/trainer-yaml-wiring.md` -- completed: trainer.yaml verification
- `graphids/config/pipeline.yaml` -- DAG topology + identity_keys
- `graphids/config/resources.yaml` -- SLURM resource profiles
