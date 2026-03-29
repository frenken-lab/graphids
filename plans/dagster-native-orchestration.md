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

| Current file | Lines | Replacement |
|---|---|---|
| `dagster_defs.py` | 513 | `component.py` (~80 lines) + `defs.yaml` (~15 lines) |
| `slurm.py` | 105 | `dagster-slurm` ComputeResource (SSH-to-localhost) |
| `resources.py` | 78 | `extra_slurm_opts` from resources.yaml, read in Component |
| `__main__.py` | 65 | `dg launch` / `dg dev` (CLI provided by dagster) |
| `config/run_dir()` | 5 | IOManager or Component computes path |
| **Total custom** | **766** | **~100 lines + YAML** |

### What stays

- `pipeline.yaml` — topology (unchanged, now the ONLY topology source)
- `ablation.yaml` — recipe (unchanged)
- `resources.yaml` — SLURM profiles (read by Component, not a separate module)
- `stages/*.yaml` + `overlays/*.yaml` — Lightning configs (unchanged)
- `run_ablation.sh` — updated to use `dg launch` instead of raw dagster CLI

## Implementation

### Phase 1: dagster-slurm spike

1. Configure `ComputeResource` with SSH-to-localhost + `skip_payload_upload=True`
2. Write one autoencoder asset using `ComputeResource.run()`
3. Submit on gpudebug, verify: sbatch → poll → COMPLETED → metadata reported
4. Compare with current `slurm.py` path — same behavior, less code

### Phase 2: CheckpointIOManager

1. Write `CheckpointIOManager` that resolves `{run_dir}/checkpoints/best_model.ckpt`
2. Wire autoencoder → curriculum with IOManager checkpoint handoff
3. Verify curriculum receives correct upstream checkpoint path

### Phase 3: SlurmTrainingComponent

1. `dg scaffold component SlurmTrainingComponent`
2. `build_defs()` reads pipeline.yaml + ablation.yaml
3. Generates AssetSpec per unique stage (deps from pipeline.yaml, not re-derived)
4. Test: `dg list defs` shows 32 assets with correct deps
5. Test: `dg launch --assets autoencoder_*` on gpudebug

### Phase 4: Delete custom code

1. Delete `dagster_defs.py`, `slurm.py`, `resources.py`, `__main__.py`
2. Remove `run_dir()`, `_CKPT_MODEL` exports from `config/__init__.py`
3. Update `run_ablation.sh` to use `dg launch`
4. Verify full ablation: smoke on gpudebug, then submit Run 005

## Prerequisite

Run 004 must complete first (or be abandoned). This redesign changes the submission
path — don't do it while debugging a failed run. Current status: Run 004 failed,
P0-P2 fixes applied, awaiting resubmission.

**Decision point:** resubmit Run 004 with current (fragile) code, or skip straight
to this redesign and submit as Run 005. The latter is cleaner but delays results.

## Risks

| Risk | Mitigation |
|---|---|
| dagster-slurm doesn't support direct-on-cluster sbatch | Phase 1 spike tests this first. Fallback: keep slurm.py |
| IOManager adds complexity for simple path passing | Phase 2 spike. Fallback: `deps=` with explicit path in metadata |
| Component system too rigid for ablation sweep | Phase 3 spike. Fallback: pure Python asset factory (current pattern, but reading pipeline.yaml) |
| PyG/torch import at dagster definition time | Component uses lazy imports; topology is pure YAML |
