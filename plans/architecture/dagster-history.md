# Dagster Orchestration — History & Lessons

> Archive of dagster-integration.md + orchestrate-rewrite.md + dagster-ablation-postmortem.md
> Active plan: `dagster-native-orchestration.md`

## Timeline

| Date | Event | Outcome |
|------|-------|---------|
| 2026-03-28 | Phase A: Evaluated dagster-slurm | Rejected (SSH claim — later found incorrect) |
| 2026-03-28 | Phase B: slurm.py + dagster_defs.py spike | gpudebug job 46121143 COMPLETED |
| 2026-03-28 | trainer.yaml wired as default_config_files | All 4 stages get callbacks, mixed precision |
| 2026-03-29 | Phase C: expand.py + ablation.yaml | 64 expanded YAMLs, 32 unique assets |
| 2026-03-29 | Phase D: Dynamic asset factory from manifest | dagster_defs.py 175 lines, dry-run success |
| 2026-03-29 | Phase E: Run 004 submitted | **100% failure** — 0 of 8 root stages completed |
| 2026-03-29 | Postmortem: 3 root causes found, all fixed | P0-P2 applied, configs regenerated |
| 2026-03-29 | P2.5: Collapsed expand.py into dagster_defs | expand.py deleted, recipe-direct loading |
| 2026-03-29 | Diagnosed: custom code reimplements dagster | dagster-native redesign plan written |

## Execution model (still valid)

Single 24-hour CPU job runs `dagster asset materialize` on Pitzer login node.
It submits GPU jobs via sbatch, polls via sacct. No daemon, no webserver.
`multiprocess_executor(max_concurrent=8)` fans out independent assets.
Restart-safe: `best_model.ckpt` check skips completed stages.

## Dagster vs Lightning responsibility split (still valid)

- **Dagster** owns: DAG ordering, partitions, retry (OOM→2x mem), skip-if-done, metrics
- **Lightning** owns: model init, training loop, checkpointing, wall-time requeue (USR1)
- **Contract:** sbatch script (`_preamble.sh` + `python -m graphids fit --config ...` + `_epilog.sh`)
- **Coupling point:** filesystem path convention (`{lake_root}/.../seed_{N}/`)

## Run 004 postmortem — key lessons

### Root causes (all fixed P0-P2)

1. **`--print_config` null serialization**: `Optional[X]=None` emits `null` in YAML,
   overriding Python defaults. Fix: explicit values in stage YAMLs.
2. **LearningRateMonitor + logger:false**: Lightning raises at `on_train_start`.
   Fix: removed LRM from trainer.yaml callbacks.
3. **`pool_aggrs=None` in GATWithJK**: No None guard at `len(pool_aggrs)`.
   Fix: `pool_aggrs = pool_aggrs or ("mean",)`.

### Process failures

- Spike found a **class of bug** (null serialization) but only fixed **instances**
- No test exercises the real config→CLI→training path
- Spike findings were never encoded as tests
- `expand.py` regenerated from source, reintroducing bugs the spike had fixed locally

### Test gaps (still open — P3)

- `test_trainer_yaml_callbacks_compatible`
- `test_expanded_configs_parse` (now: `test_recipe_configs_parse`)
- `test_gatwjk_pool_aggrs_none`
- `test_dagster_defs_load`

## Decisions made and rationale

| Decision | Rationale | Status |
|----------|-----------|--------|
| No dagster daemon/webserver | Batch execution via `dagster asset materialize` is sufficient | Valid |
| `MultiPartitionsDefinition(dataset, seed)` | Covers sweep matrix without custom code | Valid |
| dagster-slurm rejected | "Requires SSH" — **incorrect**, needs re-evaluation | **Reversed** |
| Pipes protocol rejected | Metrics from disk post-hoc is simpler | Revisit with dagster-slurm |
| Custom IOManager rejected | "Wrong abstraction for skip-if-done" — was evaluating wrong use case | **Reversed** — IOManager is for checkpoint handoff |
| expand.py for config serialization | Eliminated in P2.5, recipe-direct loading | **Superseded** |
| Identity hash from recipe types | Convention-based agreement, fragile | **Replace** with Component |

## Config system (reference)

jsonargparse + flat YAML. CLI: `python -m graphids fit --config stages/X.yaml --config overlays/Y.yaml --model.init_args.foo=bar`. Config merging, default resolution, type checking all handled by jsonargparse. See `flatten-model-config.md`.

## Ablation recipe format (reference)

`graphids/config/ablation.yaml` — 18 configs covering 6 paper claims. See file directly.
Sweep: 2 datasets x 1 seed. 32 unique assets (6 autoencoders, 8 curricula, 3 normals, 15 fusions).
