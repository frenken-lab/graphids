# Dagster-Native Orchestration Redesign

> Status: **proposed** | Date: 2026-03-29

## Problem

`dagster_defs.py` (513 lines) is a custom orchestration layer written *inside* dagster
rather than using dagster's facilities. It reimplements asset factories, config resolution,
checkpoint wiring, and SLURM submission in ad-hoc Python that must stay in sync with
`pipeline.yaml`, overlay naming conventions, model `__init__` signatures, and the recipe
format. Any change to one breaks the others silently.

Specific fragilities:
1. `_stage_args()` — 40-line if/elif chain that hardcodes per-stage config logic
2. `load_recipe()` — 60-line topology re-derivation from YAML that `pipeline.yaml` already declares
3. `_resolve_upstream_ckpts()` — manual checkpoint path wiring between stages
4. `_identity_cfg` — convention-based identity hash agreement (recipe types must match CLI-parsed types)
5. `slurm.py` — 105-line hand-rolled sbatch/sacct when `dagster-slurm` exists

## Root cause

Throughout the dagster build, the `/dagster-expert` skill and context7 documentation
were never consulted. Every capability was assumed custom, leading to ~600 lines of
orchestration code that reimplements what dagster + dagster-slurm provide natively.

## Research findings (2026-03-29)

### dagster-slurm ComputeResource

`dagster-slurm` (`ascii-supply-networks/dagster-slurm`, dagster 1.12 compat) provides:

| Feature | dagster-slurm | Our custom code |
|---------|--------------|-----------------|
| sbatch submission | `ComputeResource.run()` | `slurm.py:submit()` (25 lines) |
| sacct polling | Built into `ComputeResource` | `slurm.py:poll()` (25 lines) |
| Script generation | `BashLauncher` with `payload_path` | `slurm.py:generate_script()` (15 lines) |
| Resource overrides | `extra_slurm_opts` per asset | `resources.py` (78 lines) |
| Metrics reporting | `dagster_pipes.PipesContext.report_asset_materialization()` | `MaterializeResult(metadata=json.loads(...))` |
| Local/SLURM toggle | `mode="local"` vs `mode="slurm"` | `KD_GAT_DRY_RUN` env var |

**Prior rejection was wrong — twice.** Re-evaluated 2026-03-29:
1. "Requires SSH" — true, `SlurmResource.ssh` is required. But SSH-to-localhost works
   on OSC once `~/.ssh/id_ed25519.pub` is in `~/.ssh/authorized_keys`. The original
   evaluation (2026-03-28) and re-evaluation both failed because neither tested the
   prerequisite (`authorized_keys` was empty). Fixed: `cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys`.
2. "pixi-centric" — bypassed with `default_skip_payload_upload=True`. `BashLauncher`
   runs arbitrary scripts.
3. "Alpha maturity" — acceptable. Our custom slurm.py is also alpha-quality.

**dagster-slurm works on OSC** with SSH-to-localhost + skip_payload_upload.

**Action:** Spike one autoencoder asset with `ComputeResource` on gpudebug.

### dagster Component system

Dagster 1.9+ Components (`dg.Component` + `dg.Resolvable` + `dg.Model`) provide:

- **YAML-driven asset definitions** — `build_defs()` returns `dg.Definitions` from config
- **Template variables** — `{{ env.VAR }}`, custom `@template_var` for dynamic values
- **`dg.ResolvedAssetSpec`** — asset key/deps/groups from YAML, auto-resolved
- **Cross-component deps** — reference upstream assets by key, dagster resolves at load time
- **Scaffolding** — `dg scaffold component SlurmTrainingComponent`

This replaces `_stage_args()`, `load_recipe()`, and the asset factory loop.

### dagster IOManager

`ConfigurableIOManager` with `handle_output` / `load_input`:

- Upstream asset returns checkpoint path
- IOManager writes path to known location
- Downstream asset receives it as a function parameter
- Partitioned IOManager uses `context.asset_partition_key`

This replaces `_CKPT_OVERRIDES`, `_resolve_upstream_ckpts()`, and the manual run_dir
agreement between dagster and Lightning.

### dagster Config

`dg.Config` subclass provides typed, validated per-asset configuration. Replaces
building CLI override strings manually in `_make_asset`.

## Design

### Single source of truth: pipeline.yaml

`pipeline.yaml` already declares stages, dependencies, identity_keys, and model types.
The Component reads this directly. No Python re-derivation.

### Architecture

```
pipeline.yaml (topology)     ─┐
ablation.yaml (sweep recipe)  ├─→ SlurmTrainingComponent.build_defs()
resources.yaml (SLURM specs)  │     → AssetSpec per stage (deps from pipeline.yaml)
stages/*.yaml (Lightning cfg) ─┘     → @multi_asset with ComputeResource

ComputeResource (dagster-slurm)  ── sbatch, poll, retry
CheckpointIOManager              ── checkpoint path handoff between stages
```

### Component YAML (replaces load_recipe + _stage_args)

```yaml
# graphids/orchestrate/defs.yaml
type: graphids.orchestrate.component.SlurmTrainingComponent

attributes:
  pipeline: graphids/config/pipeline.yaml
  recipe: graphids/config/ablation.yaml
  resources: graphids/config/resources.yaml
  lake_root: "{{ env.KD_GAT_LAKE_ROOT }}"
  user: "{{ env.USER }}"
```

The Component's `build_defs()`:
1. Reads `pipeline.yaml` for stages + deps + identity_keys
2. Reads `ablation.yaml` for sweep dimensions + config overrides
3. Generates one `AssetSpec` per unique (stage, identity_hash) — deps from pipeline.yaml
4. Wraps each in a `@multi_asset` that calls `ComputeResource.run()`
5. Returns `dg.Definitions` — dagster handles the rest

### CheckpointIOManager (replaces _resolve_upstream_ckpts)

```python
class CheckpointIOManager(dg.ConfigurableIOManager):
    lake_root: str

    def handle_output(self, context: OutputContext, ckpt_path: str):
        # Upstream asset returns the checkpoint path as a string
        # IOManager just persists the path for downstream lookup
        context.add_output_metadata({"checkpoint_path": ckpt_path})

    def load_input(self, context: InputContext) -> str:
        # Returns the checkpoint path for downstream to pass as CLI arg
        upstream_run_dir = ...  # computed from upstream asset metadata
        return f"{upstream_run_dir}/checkpoints/best_model.ckpt"
```

### What gets deleted

| Current code | Lines | Replacement |
|---|---|---|
| `dagster_defs.py` `_stage_args()` | 55 | Convention-based config file resolution from pipeline.yaml |
| `dagster_defs.py` `load_recipe()` | 70 | Direct pipeline.yaml reads — topology is declared, not derived |
| `dagster_defs.py` `_resolve_upstream_ckpts()` | 15 | CheckpointIOManager |
| `dagster_defs.py` `_make_asset()` closure factory | 70 | Clean asset factory or Component `build_defs()` |
| `dagster_defs.py` `_build_assets()` dep wiring | 40 | `deps=` from pipeline.yaml `STAGE_DEPENDENCIES` directly |
| `dagster_defs.py` `smoke_test()` | 50 | Simplified — IOManager resolves ckpts, no manual wiring |
| **Total deleted** | **~300** | **~100-150 lines (factory + IOManager)** |

### What stays

- `slurm.py` — sbatch submit, sacct poll (105 lines, works fine)
- `resources.py` — ResourceSpec, get_resources, scale_resources (79 lines)
- `__main__.py` — CLI: run/validate/smoke subcommands (66 lines, may simplify)
- `pipeline.yaml` — topology (unchanged, now the ONLY topology source)
- `ablation.yaml` — recipe (unchanged)
- `resources.yaml` — SLURM profiles (unchanged)
- `stages/*.yaml` + `overlays/*.yaml` — Lightning configs (unchanged)

## Implementation

### Phase 1: Scaffold Component + CheckpointIOManager

1. `dg scaffold component SlurmTrainingComponent`
2. Write `CheckpointIOManager` — resolves `{run_dir}/checkpoints/best_model.ckpt`
3. Write `defs.yaml` pointing to pipeline.yaml, ablation.yaml, resources.yaml
4. Implement `build_defs()` — read pipeline.yaml topology, enumerate assets
5. Test: `dg list defs` shows correct assets with correct deps
6. Test: `dg check defs` passes

### Phase 2: Wire asset execution + IOManager

1. Each asset: build CLI command from stage YAML + overlay + config overrides
2. Call `slurm.py` submit/poll (retained as-is)
3. Downstream assets receive upstream ckpt path via IOManager `load_input()`
4. Port `validate_recipe()` to work with new component
5. Test: dry-run materialization produces correct sbatch commands

### Phase 3: Smoke + submit

1. Smoke test on gpudebug (one 3-stage chain, 3 epochs)
2. Verify checkpoint handoff works end-to-end across stages
3. Delete old `dagster_defs.py` code (load_recipe, _stage_args, etc.)
4. Submit Run 005 with new orchestration

## Prerequisite

Run 004 must complete first (or be abandoned). This redesign changes the submission
path — don't do it while debugging a failed run. Current status: Run 004 failed,
P0-P2 fixes applied, awaiting resubmission.

**Decision point:** resubmit Run 004 with current (fragile) code, or skip straight
to this redesign and submit as Run 005. The latter is cleaner but delays results.

## Revision: dagster-slurm dropped (2026-03-29)

### Discovery

Full API audit of `dagster-slurm` 0.x (installed, dagster 1.12.21) revealed a
**complexity mismatch** with our use case:

1. **Dagster Pipes protocol required.** `ComputeResource.run()` expects a Python
   payload script that reports back via `dagster_pipes`. Our training commands are
   `python -m graphids fit --config ...` — bash CLI commands, not Pipes-aware Python.
   Adapting requires a wrapper script that opens a Pipes session, runs subprocess,
   and reports metrics — adding indirection for no benefit.

2. **Remote-first design.** dagster-slurm's core value is pixi env packaging, SCP
   upload, and remote env extraction. We're already ON the SLURM cluster. SSH-to-
   localhost adds a needless network hop to run `sbatch` on the same machine.

3. **slurm.py is not the problem.** The 105-line `slurm.py` (generate_script,
   submit, poll) is clean, working code. The fragility lives in `dagster_defs.py`:
   `_stage_args()` if/elif chain, `load_recipe()` topology re-derivation,
   `_resolve_upstream_ckpts()` manual checkpoint wiring, and the closure-heavy
   asset factory.

### Revised scope

**Keep:** `slurm.py` (105 lines, works), `resources.py` (79 lines, works).

**Replace:** `dagster_defs.py` (513 lines) — the asset factory, recipe loading,
topology derivation, checkpoint wiring, and validate/smoke functions.

**Delete target:** ~400 lines of fragile custom code in `dagster_defs.py`, replaced
by dagster primitives (Component or asset factory reading `pipeline.yaml` directly)
+ IOManager for checkpoint handoff.

The structural win is the same: `pipeline.yaml` as single source of truth, no
re-derived topology, no manual checkpoint path wiring. We just skip the dagster-slurm
detour that would add complexity without solving the actual problem.

## Architecture decision: building blocks

> **Decision: Component** (`dg.Component` + `defs.yaml`) — decided 2026-03-29.

Dagster 1.9+ Components are the standard pattern for custom integrations. The
`build_defs()` inner logic is identical to a raw asset factory, but the Component
shell provides YAML-driven config, `dg` CLI discovery (`dg list defs`,
`dg check defs`), template variables (`{{ env.LAKE_ROOT }}`), and scaffolding
for free. Raw asset factories are the pre-1.9 pattern.

### Architecture

```
SlurmTrainingComponent (dg.Component)
├── defs.yaml — attributes: paths to pipeline.yaml, ablation.yaml, resources.yaml
├── build_defs() — reads topology from pipeline.yaml, enumerates assets
├── one @dg.asset per unique (stage, identity_hash)
│   ├── deps= from STAGE_DEPENDENCIES (pipeline.yaml)
│   ├── builds CLI command from stage YAML + overlay + overrides
│   └── calls slurm.py submit/poll
└── CheckpointIOManager — upstream ckpt path flows to downstream via IOManager

slurm.py (retained) — sbatch submit, sacct poll
resources.py (retained) — ResourceSpec, get_resources, scale_resources
```

### What the Component replaces

| Deleted code | Lines | Component equivalent |
|---|---|---|
| `_stage_args()` if/elif | 55 | Convention from pipeline.yaml: stage→config file, model_type→overlay |
| `load_recipe()` topology | 70 | `build_defs()` reads pipeline.yaml directly |
| `_resolve_upstream_ckpts()` | 15 | `CheckpointIOManager.load_input()` |
| `_make_asset()` closures | 70 | `@dg.asset` per spec, deps from pipeline.yaml |
| `_build_assets()` dep wiring | 40 | Deps declared via `AssetSpec(deps=...)` |

### Shared infrastructure (retained)

- **`slurm.py`** — sbatch submit, sacct poll (105 lines)
- **`resources.py`** — ResourceSpec, get_resources, scale_resources (79 lines)
- **`pipeline.yaml`** — single topology source
- **`ablation.yaml`** — sweep recipe
- **`resources.yaml`** — SLURM resource profiles
- **`stages/*.yaml` + `overlays/*.yaml`** — Lightning configs

## Risks

| Risk | Mitigation |
|---|---|
| IOManager adds complexity for simple path passing | Spike first. Fallback: `deps=` with explicit path in metadata |
| Component system too rigid for ablation sweep | Fallback: pure Python asset factory (Option B) |
| PyG/torch import at dagster definition time | Topology is pure YAML; lazy imports only in validate/smoke |
| dagster-slurm Pipes overhead | **RESOLVED.** Dropped dagster-slurm. Keeping slurm.py. |
