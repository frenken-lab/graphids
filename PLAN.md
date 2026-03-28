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

Spikes, Slurm logs, tests, and profiles still write to this repo and not to share lake folder

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

### Code consolidation (deferred)

- [ ] Models consolidation (`plans/models-consolidation.md`) -- registry.py dissolution, GraphModuleBase, optimizer wiring
- [ ] Preprocessing consolidation (`plans/preprocessing-consolidation.md`) -- delete _temporal.py, DataModule convention fixes

## Recently Completed

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

(none)

## Open Questions

- VGAE worker memory bloat (13G vs 22G bimodal) -- same model, different nodes. PrefetchLoader may help, needs rerun to confirm.
- `--mem` over-requesting 54G when peak is 23G -- update `resources.yaml` to 32G after confirming PrefetchLoader doesn't change RSS profile.

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
__init__.py          # package
__main__.py          # CLI entry point
resources.py         # SLURM resource profiles
submit.py            # job submission + DAG chaining
```

### Key Reference Documents

- `plans/models-consolidation.md` -- next cleanup: registry dissolution, shared base, optimizer wiring
- `plans/preprocessing-consolidation.md` -- next cleanup: delete _temporal.py, DataModule fixes
- `plans/flatten-model-config.md` -- completed config flatten reference
- `plans/tier-priority-and-implementation.md` -- priority-ordered task list
- `plans/ablation-run-001.md` -- Run 001 post-mortem
- `plans/ablation-001-training-efficiency.md` -- VRAM, GPS OOM, data staging research
- `graphids/config/pipeline.yaml` -- DAG topology + identity_keys
- `graphids/config/resources.yaml` -- SLURM resource profiles
