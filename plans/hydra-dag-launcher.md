# Hydra DAG Launcher: Custom SLURM Pipeline Sweeper with Stage Deduplication

> Researched: 2026-03-24

## Context

The KD-GAT orchestration layer currently uses two custom modules to submit multi-stage experiment DAGs to SLURM:

- **`ManifestBuilder`** (`graphids/config/manifest_builder.py`): Programmatic YAML generator with `add()`, `factorial()`, `sweep_axis()`, `write()` methods. Produces manifest files like `ablation.yaml`.
- **`submit_manifest()`** (`graphids/pipeline/orchestration/manifest.py`): Reads manifest YAML, expands sweep x configs, deduplicates stages via `identity_keys` from `pipeline.yaml`, topologically sorts the DAG, and submits via submitit with `afterok` dependencies between stages.

This works but is a **parallel system alongside Hydra**, not integrated into it. The `__main__.py` entry point has two separate codepaths: `main()` (Hydra `@hydra.main`) and `manifest()` (argparse-based). The question is whether unifying these into a single Hydra sweeper+launcher plugin is worth the effort.

## Options Considered

### Option A: Custom Hydra Sweeper Plugin (DAG-aware)

Replace `ManifestBuilder` + `submit_manifest` + the separate CLI with a Hydra sweeper plugin that:
1. Reads the manifest YAML (sweep/defaults/configs format) as its search space
2. Expands configs, deduplicates via `identity_keys`, builds DAG
3. Submits stages via the submitit launcher with `afterok` dependencies
4. Integrates with `--multirun` so the CLI is `python -m graphids --multirun hydra/sweeper=dag_pipeline manifest=ablation.yaml`

**What**: A `DagPipelineSweeper` class in `hydra_plugins/hydra_dag_pipeline/` that subclasses `Sweeper`. It implements `sweep()` by loading the manifest, computing the DAG (reusing `build_dag` logic), then calling `self.launcher.launch()` in topological order, passing SLURM dependency strings via launcher config overrides.

**Pros**:
- Single CLI entry point (`python -m graphids --multirun`) for both single-run and DAG orchestration
- Hydra manages config composition, output directories, logging, job tracking
- Plugin is reusable across projects (namespace package pattern)
- Consistent with the "3-Pillar Architecture" goal of Hydra-as-framework
- Dry-run comes free via Hydra's `--cfg job` or a custom resolver
- Manifest YAML format stays the same (backward compatible)

**Cons**:
- **Launcher API mismatch**: `Launcher.launch()` takes a flat batch `Sequence[Sequence[str]]` and returns `Sequence[JobReturn]`. It has no concept of dependencies between jobs in the batch. The sweeper would need to call `launch()` once per topological level (or once per job), threading dependency IDs between calls.
- **submitit launcher limitations**: The existing `SlurmLauncher` passes `additional_parameters` from its config, but these are static per-executor, not per-job. To set `--dependency=afterok:123` per job, you'd need to either: (a) create a new executor per job, or (b) fork/subclass the submitit launcher to accept per-job dependencies.
- **Per-job resource profiles**: The submitit launcher configures resources (GPUs, memory, partition) once per `launch()` batch. Different stages need different resources. This means one executor per resource profile, so effectively one `launch()` call per stage (or per resource group).
- **Complexity**: The sweeper needs to track job IDs returned by SLURM across `launch()` calls to build dependency chains. `launch()` returns `Sequence[JobReturn]`, but `JobReturn` doesn't expose the SLURM job ID -- the submitit `Job` object does. You'd need to either access submitit internals or write a custom launcher.
- **Testing**: Harder to unit test than a plain Python function. Hydra plugin testing requires ConfigStore setup, mock contexts, etc.
- **Effort**: Medium-large. Requires understanding Hydra internals, plugin packaging, and working around the flat-batch launcher API.

**Effort**: Large

**Sources**:
- Hydra Launcher interface: `.venv/.../hydra/plugins/launcher.py` (2 abstract methods: `setup`, `launch`)
- Hydra Sweeper interface: `.venv/.../hydra/plugins/sweeper.py` (2 abstract methods: `setup`, `sweep`)
- Hydra plugin discovery: `.venv/.../hydra/core/plugins.py` lines 56-66 (scans `hydra_plugins` namespace)
- submitit `_make_sbatch_string()`: `.venv/.../submitit/slurm/slurm.py` line 424 (native `dependency` param)
- submitit launcher: `.venv/.../hydra_plugins/hydra_submitit_launcher/submitit_launcher.py` lines 86-146 (creates one executor, calls `executor.map_array()` for entire batch)
- Hydra multirun flow: `.venv/.../hydra/_internal/hydra.py` lines 136-164 (instantiates sweeper, calls `sweeper.sweep()`)

### Option B: Custom Hydra Sweeper + Custom Launcher (full control)

Same as Option A, but also write a custom `DagSlurmLauncher` that extends `BaseSubmititLauncher` with per-job dependency and resource support.

**What**: Two plugins:
1. `DagPipelineSweeper`: Generates override batches in topological order, annotates each with resource profile and dependency info
2. `DagSlurmLauncher`: Accepts per-job metadata (dependencies, resources) alongside overrides, creates separate submitit executors per resource profile, threads SLURM job IDs for `afterok` dependencies

**Pros**:
- Clean separation: sweeper handles DAG logic, launcher handles SLURM submission
- Per-job resources and dependencies are first-class
- Could upstream the launcher as a general-purpose "DAG-aware submitit launcher"

**Cons**:
- **Two plugins to maintain** instead of one
- **Coupling**: The sweeper and launcher need a shared protocol for passing dependency/resource metadata. Hydra's `launch()` signature is `(job_overrides: Sequence[Sequence[str]], initial_job_idx: int)` -- no metadata channel. You'd need to smuggle metadata through Hydra config overrides (e.g., `hydra.launcher.dependency=afterok:123`) or a side channel.
- **More Hydra internals**: Requires deep understanding of `run_job()`, `Singleton` state management, etc.
- **Effort**: Large-to-very-large

**Effort**: Very large

**Sources**: Same as Option A, plus BasicLauncher implementation (`.venv/.../hydra/_internal/core_plugins/basic_launcher.py`)

### Option C: Keep Current Architecture, Polish It

Keep `ManifestBuilder` + `submit_manifest()` as standalone Python, but:
1. Clean up the CLI: make `manifest` a proper subcommand (e.g., `python -m graphids manifest ablation.yaml --dry-run`)
2. Add manifest validation (check identity_keys match pipeline.yaml)
3. Add progress tracking (poll SLURM job status after submission)
4. Optionally register a thin Hydra sweeper that delegates to `submit_manifest()` for `--multirun` compatibility

**What**: Keep the current working code. The "thin sweeper" variant would be a ~30-line `DagPipelineSweeper.sweep()` that calls `submit_manifest()` directly (bypassing the launcher entirely), losing Hydra's job tracking but gaining CLI unification.

**Pros**:
- **Already works.** 290 lines total, tested via dry-run, produces correct DAGs.
- **Simple to understand**: Pure Python, no plugin machinery, no Hydra internals.
- **Fast to iterate**: Change `build_dag()` or `_identity_key()` directly, no plugin reinstall.
- **submitit integration is clean**: Current code creates one `SlurmExecutor` per job, sets `dependency` directly -- no workarounds needed.
- **Effort**: Small (polish only) or medium (thin sweeper wrapper)
- **Matches code philosophy**: "Every function must earn its place." The plugin machinery doesn't earn its place if the current 290 lines work.

**Cons**:
- Two CLI entry points remain (`main()` for single-run, `manifest()` for DAG)
- Not a standard Hydra plugin -- new team members need to learn the custom manifest format
- No Hydra job tracking / output directory management for orchestrated runs
- If the thin sweeper bypasses the launcher, it loses Hydra's config composition for each job (currently done by `resolve()` in `submit_manifest()`)

**Effort**: Small (polish) / Medium (thin sweeper)

**Sources**: Current codebase files (see Source Files table below)

## Recommendation

**Option C: Keep current architecture, polish it.** Do NOT build a custom Hydra sweeper/launcher plugin.

The reasoning:

1. **The Hydra plugin API is fundamentally mismatched for DAG orchestration.** The `Launcher.launch()` interface takes a flat batch of override lists and returns results. It has no concept of inter-job dependencies, per-job resource profiles, or topological ordering. Every DAG-aware feature would require working around this API -- creating one executor per job, smuggling dependency metadata through config overrides, and accessing submitit-specific job IDs that `JobReturn` doesn't expose. This is exactly the kind of "fighting the framework" that creates maintenance burden.

2. **The current 290-line solution is simpler and more powerful than a plugin would be.** It creates one `SlurmExecutor` per job (correct for different resource profiles), passes `dependency` directly to submitit (no workaround needed), and uses `graphlib.TopologicalSorter` for ordering. A Hydra plugin would need 200+ lines of plugin boilerplate (config dataclass, ConfigStore registration, namespace package, sweeper class, potential custom launcher) to reproduce the same functionality with worse ergonomics.

3. **The manifest format is a feature, not a limitation.** The YAML manifest (`sweep`/`defaults`/`configs`) is more expressive than Hydra's override syntax for multi-stage experiments. It supports per-config stage lists, identity-key-based deduplication, and factorial expansion. Forcing this through Hydra's sweeper API would mean either losing expressiveness or reimplementing it inside the sweeper.

4. **The "two CLI entry points" problem is cosmetic.** The manifest CLI is `python -m graphids.pipeline.orchestration.manifest ablation.yaml --dry-run`. This is fine for a research project. A thin wrapper in `__main__.py` already exists as `manifest()`.

### What to polish (Option C scope)

1. **CLI cleanup**: Register `manifest` as a proper entry point in `pyproject.toml` (`[project.scripts]` section: `graphids-manifest = "graphids.__main__:manifest"`). Or add `manifest` as a subcommand via argparse in `main()`.

2. **Manifest validation**: Add a `validate_manifest()` function that checks:
   - All referenced stages exist in `pipeline.yaml`
   - All override keys are valid Hydra dotlist paths (compose a test config)
   - identity_keys referenced in pipeline.yaml exist as config fields
   - resource profiles exist for all (model, scale, stage) tuples

3. **Progress tracking**: After submission, print a table of job IDs + stages + status. Optionally poll via `sacct` or `squeue`.

4. **Export manifest YAML**: Add `--export-manifest <path>` flag to `submit_manifest` that writes the expanded DAG (with node_ids, deps, resources) to YAML for inspection/replay.

## Implementation Sketch

### 1. CLI entry point consolidation (~15 min)
Add to `pyproject.toml`:
```toml
[project.scripts]
graphids = "graphids.__main__:main"
graphids-manifest = "graphids.__main__:manifest"
```

### 2. Manifest validation (~30 min)
Add `validate_manifest(sweep, defaults, configs) -> list[str]` to `manifest.py` that returns a list of validation errors. Call it in `submit_manifest()` before `build_dag()`. Errors are actionable messages like `"Config 'kd_student' references unknown scale 'small_kd' (valid: large, small)"`.

### 3. Progress table (~20 min)
After submission, print a rich table:
```
Stage          | Dataset | Seed | Job ID  | Deps           | Status
autoencoder    | set_01  | 42   | 123456  | -              | PENDING
curriculum     | set_01  | 42   | 123457  | afterok:123456 | PENDING
...
```
Use structlog events (not rich) for SLURM-log compatibility.

### 4. Export DAG (~15 min)
Add `--export <path>` flag that writes the expanded DAG as YAML:
```yaml
jobs:
  - node_id: "autoencoder|set_01|42|..."
    stage: autoencoder
    overrides: [...]
    resources: {gpus: 1, ...}
    dep_ids: []
    config_names: [loss_x_curriculum_ce_curriculum, ...]
```

## Source Files (read during implementation)

| File | Why |
|------|-----|
| `graphids/pipeline/orchestration/manifest.py` | Main file to modify: `submit_manifest()`, `build_dag()`, add validation + export |
| `graphids/config/manifest_builder.py` | Understand manifest YAML format for validation |
| `graphids/config/pipeline.yaml` | Stage definitions, identity_keys -- validation target |
| `graphids/config/resources.yaml` | Resource profiles -- validation target |
| `graphids/config/config.yaml` | Hydra config structure for dotlist validation |
| `graphids/config/__init__.py` | `resolve()` function used to validate override keys |
| `graphids/__main__.py` | CLI entry points to consolidate |
| `pyproject.toml` | Add entry point for `graphids-manifest` |
| `tests/test_pipeline_dag.py` | Add tests for validation + export |

## Open Questions

1. **Should `graphids-manifest` be a separate CLI or a subcommand?** Separate entry point is simpler (no argparse subparsers). Subcommand is more discoverable (`graphids manifest ...`). Hydra's `@hydra.main` doesn't play well with subcommands, so a separate entry point is likely easier.

2. **Should we ever revisit the Hydra plugin approach?** Yes, if: (a) Hydra adds first-class DAG/dependency support to the Launcher API, (b) the project grows to need multi-user orchestration where Hydra's output directory conventions matter, or (c) a third-party Hydra-DAG plugin appears. None of these seem imminent.

## Cross-Repo Impact

None. The orchestration layer is internal to KD-GAT. No other repos reference `ManifestBuilder` or `submit_manifest`.
