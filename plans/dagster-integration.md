# Dagster Integration — Spec

> Status: **proposed** | Date: 2026-03-28

## Context

The ablation study requires ~60 deduplicated GPU jobs (17 configs × 2 datasets × 1 seed,
with shared upstream stages). The current `graphids/orchestrate/submit.py` handles linear
job sequences but doesn't support config expansion, stage deduplication, or the fan-out
DAG pattern. A prior Dagster integration (commit `b2e8845`, 639 lines) solved all of these
but was deleted when the config system changed. This plan restores Dagster against the
new jsonargparse + flat YAML config system.

## Execution model

**Single 24-hour CPU job as proxy daemon.** `dagster dev` runs inside an interactive or
sbatch CPU job on Pitzer. It submits GPU jobs via sbatch, polls via sacct, and handles
retry. No persistent daemon on login nodes. Safe to Ctrl+C — running GPU jobs continue
independently. Restart picks up via asset materialization checks.

```bash
sbatch --partition=cpu --time=24:00:00 --mem=4G --cpus-per-task=2 --account=PAS1266 \
  --output=slurm_logs/dagster_%j.out \
  --wrap="source scripts/slurm/_preamble.sh && dagster dev -m graphids.orchestrate.dagster_defs -p 3000"
```

Optional: tunnel port 3000 to local machine for Dagster UI visualization.

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `dagster` | Core orchestrator, asset definitions, partitions | `uv pip install dagster dagster-webserver` |
| `dagster-slurm` | SLURM submission via Dagster Pipes over NFS | `uv pip install dagster-slurm` |

**dagster-slurm** (from `ascii-supply-networks/dagster-slurm`) provides:
- `ComputeResource` with `mode="slurm"` — submits sbatch jobs per asset materialization
- `SlurmResource` + `SlurmQueueConfig` — partition, time, mem, gres config
- Dagster Pipes over NFS (`PipesFileContextInjector` / `PipesFileMessageReader`)
- Real-time log streaming back to Dagster UI
- No SSH needed (we're already on the cluster)

Evaluate whether `dagster-slurm` replaces the old hand-rolled `PipesSlurmClient` +
`pipes_slurm.py` + `slurm_primitives.py` (286 lines). If it covers our needs, delete
the custom code. If gaps exist (e.g., adaptive resource scaling on retry), wrap the
plugin rather than reimplementing.

## Components

### 1. Ablation recipe (`ablation.yaml`)

Declarative experiment spec — what to run, not how. Restored from commit `944c3ab`,
updated for the new config key names (flat, no nesting).

```yaml
sweep:
  datasets: [set_01, set_02]
  seeds: [42]

defaults:
  stages: [autoencoder, curriculum, fusion]
  scale: small
  conv_type: gatv2
  loss_fn: focal
  fusion_method: bandit
  variational: true

configs:
  # --- Loss × Curriculum factorial (claim 4) ---
  ce_normal:
    loss_fn: ce
    fusion_method: weighted_avg
    stages: [autoencoder, normal, fusion]
  ce_curriculum:
    loss_fn: ce
    fusion_method: weighted_avg
  focal_normal:
    fusion_method: weighted_avg
    stages: [autoencoder, normal, fusion]
  focal_curriculum:
    fusion_method: weighted_avg
  wce_normal:
    loss_fn: weighted_ce
    fusion_method: weighted_avg
    stages: [autoencoder, normal, fusion]
  wce_curriculum:
    loss_fn: weighted_ce
    fusion_method: weighted_avg

  # --- Fusion method (claim 2) ---
  fusion_bandit: {}
  fusion_dqn:
    fusion_method: dqn
  fusion_mlp:
    fusion_method: mlp
  fusion_weighted_avg:
    fusion_method: weighted_avg

  # --- Conv type (claim 5) ---
  conv_gatv2: {}
  conv_gatv1:
    conv_type: gat
  conv_gps:
    conv_type: gps

  # --- Unsupervised method (claim 6) ---
  unsup_vgae: {}
  unsup_gae:
    variational: false
  unsup_dgi:
    model_type: dgi
    stages: [autoencoder, normal, fusion]

  # --- Single-model baselines (claim 1) ---
  vgae_only:
    stages: [autoencoder]
  gat_only:
    stages: [normal]

  # --- KD & scale (claim 3) ---
  large_reference:
    scale: large
  kd_student:
    auxiliaries:
      - type: kd
        alpha: 0.7
        teacher_scale: large
```

### 2. Config expander

Reads `ablation.yaml` + stage/overlay YAMLs → emits per-job merged YAML configs.

**Input:** ablation recipe + `graphids/config/stages/*.yaml` + `graphids/config/overlays/*.yaml`

**Output:** one merged YAML per (config × dataset × seed × stage), with:
- Correct `model.class_path` and `data.class_path` for the stage
- Merged `model.init_args` from stage defaults + overlay + ablation overrides
- `trainer.default_root_dir` set to the identity-hashed run directory
- `seed_everything` set

**Deduplication:** Multiple ablation configs can share upstream stages. The expander
computes identity hashes and deduplicates — e.g., one VGAE autoencoder
(`vgae_small_autoencoder_{hash}`) serves many downstream GAT configs that only differ
in `loss_fn` or `fusion_method`.

**Key question:** Does this need to be a Python script that generates files, or can Dagster's
asset factory pattern handle config expansion at definition time? The old code used
`build_dag_topology()` which constructed `DagNode` objects from `PipelineConfig.variants`.
The new version would construct them from the ablation recipe.

### 3. Asset definitions (DAG)

Software-defined assets, one per unique (model_type, scale, stage, identity_hash).
Generated dynamically from the ablation recipe via an asset factory.

**Asset factory pattern:**

```
ablation.yaml
  → config expander (dedup by identity hash)
    → DagNode per unique stage
      → @dg.asset per DagNode
```

Each asset:
- Declares `deps=` pointing to upstream assets (from `STAGE_DEPENDENCIES`)
- Has `partitions_def=MultiPartitionsDefinition({"dataset": ..., "seed": ...})`
- On materialization: submits `python -m graphids fit --config <merged.yaml>` via SLURM
- Reports metrics back via Dagster Pipes

**KD dependency:** The `kd_student` config's autoencoder/curriculum assets depend on the
corresponding `large_reference` assets (teacher must complete first). The expander
encodes this as an explicit asset dependency.

### 4. SLURM resource profiles

Already exist in `graphids/config/resources.yaml` + `graphids/orchestrate/resources.py`.
Dagster assets look up `(model_type, scale, stage) → ResourceSpec` and pass to the
SLURM submission layer.

**Adaptive retry:** On `OUT_OF_MEMORY` → 2× mem, on `TIMEOUT` → 1.5× time. Already
implemented in `resources.py:scale_resources()`. Wire into Dagster's `RetryPolicy` +
`RetryRequested` pattern (same as old `dagster_defs.py:102-112`).

### 5. SLURM submission layer

**Evaluate dagster-slurm first.** The plugin provides `ComputeResource(mode="slurm")`
which handles sbatch submission, polling, and Pipes protocol. Our requirements:

| Need | dagster-slurm? | Old custom code? |
|------|---------------|-----------------|
| sbatch submission | Yes (`SlurmResource`) | `submit_sbatch()` |
| Poll until done | Yes (built-in) | `poll_until_done()` |
| Pipes over NFS | Yes (`PipesFile*`) | `PipesFileContextInjector` |
| Per-asset resource config | Yes (`extra_slurm_opts`) | `get_resources()` |
| Adaptive retry scaling | **No** — needs custom | `scale_resources()` |
| Log streaming | Yes (built-in) | Not implemented |
| `_preamble.sh` sourcing | Needs `extra_env` or wrapper | `generate_sbatch_script()` |

**Gap:** Adaptive resource scaling on retry (OOM → 2× mem) is not in dagster-slurm.
Wrap `ComputeResource.run()` to check `context.retry_number` and adjust `extra_slurm_opts`.

**Gap:** Our jobs need `source _preamble.sh` before the Python command. Either:
- Use `extra_env` to replicate what preamble sets (module load, venv, CUDA config)
- Or use a wrapper script as the `payload_path`

### 6. Pipes integration (GPU job → Dagster)

Each GPU job (the Lightning training process) reports completion metrics back via
Dagster Pipes. This is a thin wrapper at the end of `__main__.py` or in `_epilog.sh`:

```python
# At end of training, if running under Dagster Pipes:
from dagster_pipes import open_dagster_pipes
with open_dagster_pipes() as pipes:
    pipes.report_asset_materialization(metadata={...})
```

This is optional — Dagster can also detect completion from sbatch exit code alone.
Pipes adds metric reporting (val_loss, test_f1, etc.) visible in the Dagster UI.

### 7. Materialization checks (resume/skip)

Each asset checks if its output already exists before submitting:
- `best_model.ckpt` in the expected run directory → skip (already materialized)
- `last.ckpt` exists → pass `--ckpt_path last.ckpt` for Lightning auto-resume

This enables restart after Ctrl+C or daemon crash without rerunning completed stages.
Same pattern as old `dagster_defs.py` `_is_done()` check and current `submit.py:96-98`.

## File layout

```
graphids/orchestrate/
  __init__.py              # package (keep existing)
  __main__.py              # CLI entry point (keep existing, add dagster mode)
  dagster_defs.py          # Definitions: assets, resources, partitions
  config_expander.py       # ablation.yaml → per-job merged YAML + dedup
  resources.py             # ResourceSpec + scale_resources (keep existing)
  submit.py                # Legacy linear orchestrator (keep for simple runs)

ablation.yaml              # Experiment recipe (project root)
```

## What gets deleted

| File | Lines | Reason |
|------|-------|--------|
| `graphids/orchestrate/submit.py` | 246 | Replaced by Dagster assets (keep if needed for one-off runs) |

**Or keep both:** `submit.py` for quick one-off stage submissions, Dagster for full
ablation/sweep orchestration. No conflict — different entry points.

## What gets restored (from `b2e8845`, adapted)

| Old file | Lines | New location | Adaptation needed |
|----------|-------|-------------|-------------------|
| `dagster_defs.py` | 271 | `orchestrate/dagster_defs.py` | Replace `resolve()` + `PipelineConfig.variants` with ablation recipe expander |
| `pipes_slurm.py` | 123 | **Maybe deleted** — evaluate `dagster-slurm` plugin first | If plugin covers needs, delete |
| `slurm_primitives.py` | 245 | Partially in `resources.py` already | `generate_sbatch_script` + `poll_until_done` — check if dagster-slurm replaces |

## Phases

### Phase A: Install + spike (1 session)

1. `uv pip install dagster dagster-webserver dagster-slurm`
2. Minimal `dagster_defs.py`: one hardcoded asset (VGAE autoencoder, small, set_01)
3. Test `dagster dev` on CPU job — does it start, show UI, submit one GPU job?
4. Test dagster-slurm `ComputeResource(mode="slurm")` — does it handle sbatch + poll?
5. Decision: dagster-slurm vs custom PipesSlurmClient

### Phase B: Config expander + ablation recipe (1 session)

1. Write `ablation.yaml` (adapted from `944c3ab` recipe above)
2. Write `config_expander.py`: recipe → per-job merged YAMLs with dedup
3. Test: expander produces correct YAML for all 17 configs, deduped to ~25 unique stages per dataset

### Phase C: Full DAG (1 session)

1. Asset factory: expander output → `@dg.asset` per unique stage
2. Multi-partitions: `{"dataset": [...], "seed": [...]}`
3. KD dependency wiring
4. Adaptive retry via `RetryPolicy` + `scale_resources`
5. Materialization checks (skip completed)
6. Dry-run test: `dagster asset materialize --select "*"` with `dry_run=True`

### Phase D: Ablation Run 004 (execution)

1. Submit 24hr CPU job with `dagster dev`
2. Materialize all assets for `(set_01, set_02) × seed_42`
3. Monitor via Dagster UI (tunneled) or `sacct`
4. After large_reference completes → KD student assets auto-trigger
5. Eval: `python -m graphids test` per completed training run

## Open questions

1. **dagster-slurm vs custom:** Does the plugin handle our `_preamble.sh` + per-asset
   resource profiles, or do we need the custom `PipesSlurmClient`?

2. **Config expander location:** Should the expander run at Dagster definition time
   (assets are dynamic) or as a pre-step that writes files to disk?

3. **Eval as assets:** Should `python -m graphids test` be a Dagster asset downstream
   of training, or a separate post-hoc step? Asset approach gives automatic triggering
   but adds complexity.

4. **HPO integration (Phase 2):** Optuna trials as Dagster assets? Or separate Optuna
   study that submits through the same SLURM layer?
