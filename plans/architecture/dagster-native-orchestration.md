# Dagster-Native Orchestration Redesign

> Status: **implemented** | Proposed: 2026-03-29 | Audited: 2026-03-30

## Problem (original)

The original `dagster_defs.py` (~513 lines) was a custom orchestration layer written
*inside* dagster rather than using dagster's facilities. It reimplemented asset factories,
config resolution, checkpoint wiring, and SLURM submission in ad-hoc Python.

Specific fragilities: `_stage_args()` if/elif chain, `load_recipe()` topology
re-derivation, `_resolve_upstream_ckpts()` manual checkpoint wiring, closure-heavy
`_make_asset()` factory, and `_build_assets()` dep wiring.

## What was implemented

`dagster_defs.py` was deleted and replaced by a dagster Component architecture:

### Current file inventory (`graphids/orchestrate/`)

| File | Lines | Role |
|------|-------|------|
| `component.py` | 472 | Main logic: Component, IOManager, Resource, asset factory, config resolution |
| `__main__.py` | 224 | CLI: `run`, `validate`, `smoke` subcommands |
| `slurm.py` | 106 | sbatch submit, sacct poll (retained from original) |
| `resources.py` | 78 | ResourceSpec, get_resources, scale_resources (retained from original) |
| `definitions.py` | 17 | Dagster entry point — instantiates `SlurmTrainingComponent` |
| `__init__.py` | 12 | Docstring-only |
| **Total** | **909** | |

### `component.py` architecture

```
SlurmTrainingComponent (dg.Component, dg.Model, dg.Resolvable)  [line 427]
├── build_defs() — reads pipeline.yaml topology + ablation.yaml recipe
│   ├── enumerate_assets(PIPELINE_YAML, recipe) → list[StageConfig]  [line 178]
│   ├── MultiPartitionsDefinition (datasets × seeds)
│   ├── one @asset per StageConfig via _make_asset()  [line 293]
│   ├── _make_checkpoint_checks() for asset health  [line 381]
│   └── returns dg.Definitions with SlurmTrainingResource + CheckpointPathIOManager
│
├── CheckpointPathIOManager (dg.ConfigurableIOManager)  [line 57]
│   ├── handle_output — writes ckpt path string to JSON sidecar
│   └── load_input — reads upstream sidecar, returns ckpt path for downstream
│
├── SlurmTrainingResource (dg.ConfigurableResource)  [line 87]
│   └── submit_and_wait — calls slurm.py submit() + poll()
│
└── Config resolution helpers
    ├── _resolve_config_files(stage, stage_def, merged)  [line 129]
    ├── _overlay_model(stage_def, merged)  [line 120]
    ├── _identity_value(key, merged, stages)  [line 148]
    └── StageConfig (frozen dataclass)  [line 162]
```

Key constants:
- `_CKPT_FLAG` (line 42-46) — maps model type → CLI flag for ckpt path (`vgae` → `--data.init_args.vgae_ckpt_path`)
- `_RECIPE_TO_IDENTITY` (line 49) — maps recipe keys to identity keys where names differ
- `RECIPE_PATH` (line 39) — defaults to `recipes/ablation.yaml`, overridden by `KD_GAT_RECIPE` env var

### `definitions.py` entry point

```python
component = SlurmTrainingComponent(
    dry_run=os.environ.get("KD_GAT_DRY_RUN", "").lower() in ("1", "true"),
)
defs = build_defs_for_component(component)
```

Discovered by `dg` CLI via `pyproject.toml` `code_location_target_module`.

### `__main__.py` subcommands

| Subcommand | Handler | Lines |
|------------|---------|-------|
| `run` | Calls `dg launch` via subprocess | 197-208 |
| `validate` | `validate_recipe()` — checks config chains parse, no incompatibilities | 33-88 |
| `smoke` | `smoke_test()` — runs a 3-stage chain on gpudebug | 91-176 |

### `slurm.py` (retained)

| Function | Lines | Signature |
|----------|-------|-----------|
| `generate_script` | 26-43 | `(config_files, resources, *, ckpt_path, cli_overrides) → str` |
| `submit` | 46-76 | `(script, resources, *, job_name, dry_run) → int` |
| `poll` | 79-106 | `(job_id, *, interval, max_unknown) → str` |

### `resources.py` (retained)

| Symbol | Lines | Notes |
|--------|-------|-------|
| `ResourceSpec` | 13-32 | Dataclass: partition, time, mem, cpus_per_task, num_workers, gres |
| `get_resources` | 39-50 | `(model_type, scale, stage) → ResourceSpec` |
| `get_failure_reactions` | 53-54 | Not in original plan — failure reaction lookup |
| `scale_resources` | 57-78 | `(spec, failure_reason) → ResourceSpec` |

## What changed vs the original plan

### Implemented as proposed

- `pipeline.yaml` as single source of truth — `enumerate_assets()` reads topology directly
- `CheckpointPathIOManager` — ckpt path handoff via JSON sidecars (no manual wiring)
- `SlurmTrainingComponent` as `dg.Component` — YAML-like config, `dg` CLI discovery
- `slurm.py` and `resources.py` retained unchanged
- `dagster-slurm` dropped for SLURM submission (custom `SlurmTrainingResource` instead)

### Deviated from plan

| Planned | Actual | Reason |
|---------|--------|--------|
| `defs.yaml` for Component config | Python-based `definitions.py` (17 lines) | Simpler; env var logic is one line of Python |
| `~66 lines` for `__main__.py` | 224 lines | `validate_recipe()` (56 lines) and `smoke_test()` (86 lines) are substantial |
| `~100-150 lines` replacement code | 472 lines in `component.py` alone | `enumerate_assets()` (106 lines) and `_make_asset()` (81 lines) are non-trivial |
| `_stage_args()` replaced by "convention" | `_resolve_config_files()` + `_overlay_model()` (24 lines total) | Clean, but still Python logic |

### Net line count

| Before | After |
|--------|-------|
| `dagster_defs.py` ~513 lines (single file) | `component.py` 472 + `definitions.py` 17 = 489 lines |
| `__main__.py` (unknown, existed before) | `__main__.py` 224 lines |
| `slurm.py` ~105 | `slurm.py` 106 (unchanged) |
| `resources.py` ~79 | `resources.py` 78 (unchanged) |

The redesign restructured rather than reduced — complexity moved from ad-hoc code to
dagster-native patterns (Component, IOManager, Resource), but total line count is similar.

## Remaining issues

### 1. `dagster-slurm` unused dependency

`dagster-slurm>=1.12.0` is declared in `pyproject.toml` (line 34) and installed in `.venv`,
but **zero Python files import it**. `SlurmTrainingResource` in `component.py` is a custom
`ConfigurableResource`, not from `dagster-slurm`. The dependency should be removed from
`pyproject.toml` to avoid confusion and reduce install weight.

### 2. No `defs.yaml`

The plan proposed YAML-driven component configuration via `defs.yaml` with template
variables (`{{ env.KD_GAT_LAKE_ROOT }}`). The actual implementation uses Python-based
config in `definitions.py`. This is simpler but means `dg scaffold` and `dg list components`
won't discover the component via YAML. Acceptable trade-off for current scale.

### 3. Dagster Pipes not adopted

The plan's research section discussed `dagster_pipes.PipesContext` for metrics reporting.
This was not implemented — metrics flow via `MaterializeResult(metadata=...)` in
`_make_asset()`. No `dagster_pipes` import exists in the codebase. Fine for now; Pipes
would only add value for cross-process metric streaming.

## dagster-slurm evaluation (historical)

> Preserved for decision context. dagster-slurm was evaluated and dropped.

`dagster-slurm` 0.x (now 1.12.0, dagster 1.12.21 compat) provides `ComputeResource` with
sbatch/sacct, `BashLauncher`, and local/SLURM mode toggle. Evaluated 2026-03-28/29.

**Rejected because:**
1. **Dagster Pipes protocol required.** `ComputeResource.run()` expects Pipes-aware Python
   payloads. Our training commands are `python -m graphids fit --config ...` — bash CLI
   commands. Adapting requires a wrapper script with no benefit.
2. **Remote-first design.** dagster-slurm's core value is pixi env packaging + SCP upload.
   We're already ON the SLURM cluster.
3. **slurm.py is not the problem.** The 106-line `slurm.py` is clean, working code. The
   fragility was in the asset factory / config resolution layer.

SSH-to-localhost prerequisite was verified working (`~/.ssh/authorized_keys` has key),
but the complexity mismatch made adoption net-negative.

## Risks (current)

| Risk | Status |
|------|--------|
| IOManager adds complexity for simple path passing | Implemented; JSON sidecars work |
| Component system too rigid for ablation sweep | Working; `enumerate_assets()` handles recipe expansion |
| PyG/torch import at dagster definition time | Solved — `component.py:8` says "NO torch/Lightning imports at definition time" |
| dagster-slurm Pipes overhead | N/A — dagster-slurm dropped |
| Unused dagster-slurm dep in pyproject.toml | **Open** — should remove |
