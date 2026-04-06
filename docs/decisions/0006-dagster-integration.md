# Dagster Orchestration — History & Lessons

> **Stale reference note (2026-04-06):** `dagster_defs.py` -> `graphids/orchestrate/definitions.py`. `orchestrate/assets.py` -> `orchestrate/dagster/assets.py`. `orchestrate/checks.py` -> `orchestrate/dagster/checks.py`. `slurm/slurm.py` -> `slurm/pipeline.py`. `python -m graphids.orchestrate run` no longer valid -- CLI uses `graphids/commands/` entry points.

> Consolidated from dagster-integration.md + orchestrate-rewrite.md + dagster-ablation-postmortem.md
> Current architecture: `dagster-native-orchestration.md`
> Audited: 2026-03-31

## Timeline

| Date | Event | Outcome |
|------|-------|---------|
| 03-27 | Wire LightningCLI config, add orchestrator + resource profiles | Initial `dagster_defs.py` |
| 03-28 | Flatten all LightningModule configs to typed primitives | Config owned by `config-system.md` rules file |
| 03-28 | Evaluate dagster-slurm: rejected | Custom `SlurmTrainingResource` used instead |
| 03-28 | trainer.yaml wired as default_config_files | All stages get callbacks, mixed precision |
| 03-29 | **Rebuild with proper primitives** | component.py + definitions.py. IOManager, Resource, Component. |
| 03-29 | Fix 8 runtime bugs from smoke test | lake_root, SaveConfigCallback, CurriculumDM, fusion routing, etc. |
| 03-30 | **Run 004** (ablation, set_01/set_02) | 0/36 completed across 2 attempts |
| 03-30 | Fix Run 004: RAM profiles, dagster logging, VRAM probe, observability | All 6 issues resolved |
| 03-31 | **Run 005** (ablation + main_results, Ascend A100) | 22/36 completed. Post-mortem deleted 2026-04-02 (fixes applied, see git history) |
| 03-31 | Fix Run 005: fusion wiring, zero-copy preprocessing, wall times | All 3 fixes applied |

## Execution Model

Single CPU SLURM job runs dagster orchestrator. Submits GPU training jobs via sbatch, polls via sacct.
- Entry: `python -m graphids.orchestrate run --recipe ...`
- `SlurmTrainingComponent.build_defs()` generates assets from `pipeline.yaml` + recipe
- `multiprocess_executor(max_concurrent=8)` fans out independent assets
- Restart-safe: skip requires both `best_model.ckpt` AND `.complete` marker
- `DAGSTER_HOME=/fs/scratch/PAS1266/dagster` (SQLite run history)

## Dagster vs Lightning Responsibility Split

| Layer | Owns | Coupling |
|-------|------|----------|
| **Dagster** | DAG ordering, partitions (dataset×seed), retry, skip-if-done, metadata | Filesystem path convention |
| **Lightning** | Model init, training loop, checkpointing, wall-time requeue (USR1) | `python -m graphids fit --config ...` |
| **IOManager** | Checkpoint path handoff between stages via JSON sidecars | Upstream asset returns ckpt path string |

## Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| dagster-slurm rejected | Pipes protocol overhead, remote-first design, slurm.py not the problem | Confirmed. Unused dep in pyproject.toml — tracked in `open_issues.md` |
| Dagster Pipes rejected | Training is CLI commands, not Pipes-aware Python | Valid — revisit if in-job metric streaming needed |
| Custom CheckpointPathIOManager | JSON sidecar for ckpt path handoff | Implemented |
| dagster Component (`dg.Component`) | YAML-driven config, `dg` CLI discovery | Implemented |
| .complete marker for skip-if-done | `best_model.ckpt` alone insufficient (stale/killed runs) | Implemented |

## Pre-submission Bugs (caught before Run 004)

1. `--print_config` null serialization: `Optional[X]=None` → `null` overrides Python defaults. Fix: explicit values in stage YAMLs.
2. LearningRateMonitor + `logger:false`: Lightning raises at `on_train_start`. Fix: removed LRM.
3. `pool_aggrs=None` in GATWithJK: no None guard at `len()`. Fix: `pool_aggrs or ("mean",)`.

## Cross-references

- Architecture: `dagster-native-orchestration.md`
- Open test gaps + observability items: `../backlog/open-items.md`
- Run post-mortems: deleted 2026-04-02 (all fixes applied, value captured in `../backlog/open-items.md`)
