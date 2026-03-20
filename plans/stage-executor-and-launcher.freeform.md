# Stage Executor + Launcher Protocol

**Date:** 2026-03-20
**Status:** Brainstorm
**Depends on:** `codebase-reduction.md` (investigation), `pipeline-toolchain-decision.md` (prior decision — partially superseded)

---

## Problem

Pipeline has 4 entry paths with inconsistent guarantees:

| Path | Entry | Validates? | Manifest? | Logging? | Archive? |
|------|-------|-----------|-----------|----------|----------|
| CLI | `cli.py → _run_single_stage()` | Yes | Yes | Yes | Yes |
| API | `api.py → STAGE_FNS[stage](cfg)` | **No** | **No** | **No** | **No** |
| Fire-and-forget | SLURM → CLI (Path 1 inside job) | Yes | Yes | Yes | Yes |
| Dagster daemon | SLURM → CLI + Pipes | Yes | Yes | Yes | Yes |

Root cause: cross-cutting concerns live in `cli.py:_run_single_stage()` — a CLI-specific function. `api.py` bypasses it. Future backends (submitit, K8s) would need to duplicate it or also bypass.

Second problem: platform assumptions (SLURM) leak through 4+ files. Can't swap backend without touching `dagster_defs.py`, `pipes_slurm.py`, `slurm_primitives.py`, `subprocess_utils.py`.

## Goals

1. **Single executor**: every path through the pipeline gets validation, manifest, logging, archive
2. **Platform-agnostic orchestration**: DAG logic, retry policy, resource specs — no SLURM imports
3. **Thin backend protocol**: only "submit/status/result/cancel" is platform-specific
4. **Drop Dagster dependency**: not using daemon mode; SLURM-native patterns + Launcher protocol cover all current needs
5. **submitit for SLURM backend**: Python callable → SLURM job (no bash script generation)

## Non-goals

- Multi-cluster federation
- Real-time streaming metrics (manifest is batch)
- Web UI (headless HPC)

---

## Architecture

```
 Entry points (thin)          Custom code (domain-specific)     Stdlib / libraries
┌──────────┐                 ┌─────────────────────┐          ┌───────────────────────┐
│ cli.py   │─── resolve ───→│ execute_stage()      │          │ concurrent.futures     │
│ api.py   │─── resolve ───→│   ~40 lines          │          │   .Executor (protocol) │
│ notebook │─── resolve ───→│   validate, log,     │          │   .Future              │
└──────────┘                 │   archive, manifest  │          │   .ProcessPoolExecutor │
                             └─────────────────────┘          ├───────────────────────┤
┌──────────┐                 ┌─────────────────────┐          │ submitit               │
│ run_dag()│── submit ──────→│ make_slurm_executor()│─────────→│   .SlurmExecutor       │
│ ~30 lines│  (futures)      │   ~40 lines          │          │   (IS an Executor)     │
│ graphlib │                 │   ResourceSpec →     │          ├───────────────────────┤
│ + futures│                 │   submitit kwargs    │          │ graphlib               │
└──────────┘                 └─────────────────────┘          │   .TopologicalSorter   │
                                                              └───────────────────────┘
                             ~110 lines custom total
                             Everything else is stdlib or submitit
```

---

## Component specs

### 1. `execute_stage(cfg, stage) -> StageResult`

**Location:** `graphids/pipeline/executor.py`
**Owns:** all cross-cutting concerns (currently in `cli.py:_run_single_stage`)

```python
@dataclass(frozen=True)
class StageResult:
    metrics: dict[str, float]
    duration_seconds: float
    checkpoint_path: Path | None
    manifest_path: Path

def execute_stage(cfg: PipelineConfig, stage: str) -> StageResult:
    """Single entry point for ALL stage execution."""
    # 1. Logging (idempotent — safe to call multiple times)
    configure_logging()
    bind_contextvars(dataset=cfg.dataset, model=cfg.model_type,
                     scale=cfg.scale, stage=stage, seed=cfg.seed)

    # 2. Validate
    validate(cfg, stage)

    # 3. Gateway (single instance)
    gw, mapper = open_gateway(cfg)
    sdir = gw.resolve(stage)

    # 4. Archive previous run
    archive = _archive_previous(sdir)

    # 5. Config snapshot
    mapper.save_config(cfg, stage)

    # 6. Run
    t0 = time.monotonic()
    try:
        result = STAGE_FNS[stage](cfg)
        duration = time.monotonic() - t0
        metrics = _extract_metrics(result, duration)

        # 7. Manifest
        manifest_path = write_manifest(sdir, cfg, stage, metrics)

        # 8. Pipes reporting (auto-detected, zero-cost if not under Dagster)
        _report_pipes(metrics)

        # 9. Cleanup archive
        _delete_archive(archive)

        return StageResult(metrics, duration, _find_checkpoint(sdir), manifest_path)

    except Exception:
        _restore_archive(archive, sdir)
        raise
```

**Key change from current:** stage functions no longer create their own gateway. `_save_and_cleanup()` in training.py uses `open_gateway(cfg)` internally — this gets refactored so the mapper is either passed in or stage functions just return the model and let the executor save.

**Decision needed:** do stage fns receive mapper as arg, or keep self-contained? Tradeoff:
- Injected mapper → cleaner I/O boundary, testable, but changes all stage fn signatures
- Self-contained → less churn, but two gateways persist (executor + stage fn each create one)

Recommendation: **injected mapper** for new code, but `open_gateway(cfg)` returns the same instance for the same cfg (flyweight), so the "two gateway" problem is solved without signature changes. Add `@lru_cache` or module-level registry keyed on `(lake_root, dataset, model_type, scale, seed)`.

### 2. No custom Launcher — use `concurrent.futures.Executor`

`concurrent.futures.Executor` IS the launcher protocol. Already in stdlib:
```python
class Executor:
    def submit(fn, *args, **kwargs) -> Future: ...
    # Future: .result(), .done(), .cancel(), .exception()
```

Both backends already implement it:
- **SLURM**: `submitit.SlurmExecutor(Executor)` — submit Python callables to SLURM
- **Local**: `concurrent.futures.ProcessPoolExecutor` — stdlib
- **Future K8s/cloud**: write one `Executor` subclass, same interface

No `launcher.py`. No `JobHandle`. No `LocalLauncher`. These are stdlib.

### 3. `make_slurm_executor(resources) -> submitit.SlurmExecutor`

**Location:** `graphids/pipeline/orchestration/slurm.py` (~40 lines)

Not a class — a factory function that maps `ResourceSpec` → submitit kwargs:

```python
def make_slurm_executor(
    resources: ResourceSpec,
    *,
    depends_on: list[str] = [],  # SLURM job IDs
    setup: list[str] = ["source scripts/slurm/_preamble.sh"],
) -> submitit.SlurmExecutor:
    """Configure a submitit executor from ResourceSpec."""
    executor = submitit.SlurmExecutor(folder="slurm_logs")
    extra = {"signal": "B:USR1@180"}
    if depends_on:
        extra["dependency"] = f"afterok:{':'.join(depends_on)}"

    executor.update_parameters(
        mem_gb=resources.memory_gb,
        gpus_per_node=resources.gpus,
        cpus_per_task=resources.cpus,
        timeout_min=int(resources.walltime.total_seconds() / 60),
        slurm_partition=resources.partition,
        slurm_account=SLURM_ACCOUNT,
        setup=setup,
        slurm_additional_parameters=extra,
    )
    return executor
```

Usage: `executor = make_slurm_executor(resources); job = executor.submit(execute_stage, cfg, stage)`

PipelineConfig is frozen Pydantic → picklable. submitit pickles fn + args, unpickles on compute node. **No CLI string round-trip.** No `build_cli_cmd()`. No `generate_sbatch_script()`.

### 4. `run_dag()` — not a file, a function

**Location:** lives in `graphids/pipeline/orchestration/__init__.py` or `dag.py` (~30 lines)

Uses `graphlib.TopologicalSorter` (stdlib) + `concurrent.futures.Future`:

```python
def run_dag(
    executor_factory,   # Callable[[ResourceSpec, list[str]], Executor]
    dag: dict[str, DagNode],
    dataset: str,
    seeds: list[int],
) -> dict[str, Future]:
    """Submit DAG through any concurrent.futures.Executor."""
    topo_order = list(graphlib.TopologicalSorter(
        {name: set(node.deps) for name, node in dag.items()}
    ).static_order())

    futures: dict[str, Future] = {}
    for seed in seeds:
        for name in topo_order:
            node = dag[name]
            # Resolve deps to job IDs (for SLURM --dependency)
            dep_ids = [futures[(d, seed)].job_id for d in node.deps
                       if (d, seed) in futures]
            resources = get_resources(node.resource_model, node.scale, node.stage)
            cfg = resolve(node.cli_model, node.scale, dataset=dataset, seed=seed)

            executor = executor_factory(resources, dep_ids)
            futures[(name, seed)] = executor.submit(execute_stage, cfg, node.stage)

    return futures
```

**Local mode**: `executor_factory = lambda res, deps: ProcessPoolExecutor(1)`
**SLURM mode**: `executor_factory = lambda res, deps: make_slurm_executor(res, depends_on=deps)`

Retry is a wrapper (~15 lines) that catches `submitit.SlurmJobFailed`, calls `scale_resources()`, resubmits.

---

## What gets deleted

| File | Lines | Reason |
|------|------:|--------|
| `subprocess_utils.py` | 76 | `build_cli_cmd()` — no CLI round-trip with submitit |
| `slurm_primitives.py` | 245 | `generate_sbatch_script()`, `submit_sbatch()`, `poll_until_done()` → SlurmLauncher |
| `pipes_slurm.py` | 124 | `PipesSlurmClient` — Dagster Pipes not needed |
| `dagster_defs.py` | 271 | `build_dagster_assets()`, Dagster `defs` — replaced by `run_dag()` |
| `job.py` (partial) | ~40 | `ResourceSpec.from_yaml()` → Pydantic validators, `mem_slurm`/`walltime_slurm` → gone |
| **Total deleted** | **~756** | |

## What gets created

| File | Est. lines | Role |
|------|------:|------|
| `pipeline/executor.py` | ~40 | `execute_stage()` + `StageResult` (domain-specific, no library replacement) |
| `pipeline/orchestration/slurm.py` | ~40 | `make_slurm_executor()` — ResourceSpec → submitit kwargs |
| `pipeline/orchestration/__init__.py` (extend) | ~30 | `run_dag()` — graphlib + concurrent.futures (no custom protocol) |
| ~~`launcher.py`~~ | 0 | `concurrent.futures.Executor` IS the protocol (stdlib) |
| ~~`backends/local.py`~~ | 0 | `concurrent.futures.ProcessPoolExecutor` (stdlib) |
| **Total created** | **~110** | |

**Net: -646 lines, plus platform portability gained.**

---

## Spike plan (before any refactor)

### Spike 1: submitit on OSC
Prove submitit can submit a Python callable with preamble, get result back.
```python
# tests/spikes/spike_submitit.py
import submitit
executor = submitit.SlurmExecutor(folder="slurm_logs/spikes")
executor.update_parameters(
    mem_gb=4, timeout_min=5, partition="serial",
    slurm_account="PAS1266",
    setup=["source scripts/slurm/_preamble.sh"],
)
def test_fn(x): return x ** 2
job = executor.submit(test_fn, 42)
print(f"Job ID: {job.job_id}")
print(f"Result: {job.result()}")  # blocks
```
**Validates:** preamble sourcing, pickle round-trip, result retrieval.

### Spike 2: PipelineConfig pickling
Prove frozen Pydantic model survives pickle round-trip (submitit uses cloudpickle).
```python
# tests/spikes/spike_pickle_config.py
import cloudpickle, pickle
from graphids.config import resolve
cfg = resolve("vgae", "large", dataset="hcrl_sa")
data = cloudpickle.dumps(cfg)
cfg2 = pickle.loads(data)
assert cfg == cfg2
assert cfg2.vgae.latent_dim == cfg.vgae.latent_dim
```

### Spike 3: submitit signal forwarding
Prove SIGUSR1 from SLURM reaches Python (Lightning auto-requeue needs this).
```python
# tests/spikes/spike_submitit_signal.py
import signal, time, submitit
def signal_test():
    received = []
    signal.signal(signal.SIGUSR1, lambda s, f: received.append("USR1"))
    time.sleep(300)  # will be killed by timeout
    return received
# Submit with 1-min timeout, --signal=B:USR1@30
# Check if "USR1" is in the result
```

### Spike 4: submitit dependency chains
Prove `--dependency=afterok:$JOB_ID` works via `slurm_additional_parameters`.

---

## Migration order

1. **Extract `execute_stage()`** from `cli.py` → `pipeline/executor.py`. Wire `cli.py` and `api.py` to call it. Zero behavioral change, all current paths get full guarantees. **No spike needed.**

2. **Run spikes 1-4.** If any fail, update this doc with why before proceeding.

3. **Write `make_slurm_executor()` + `run_dag()`.** Wire `fire_and_forget()` → `run_dag(make_slurm_executor, ...)`. Local mode uses `ProcessPoolExecutor` — no custom code.

4. **Delete old orchestration.** Remove `subprocess_utils.py`, `slurm_primitives.py`, `pipes_slurm.py`, `dagster_defs.py`. Remove `dagster`/`dagster-pipes` from deps.

5. **Simplify ResourceSpec.** Pydantic validators replace `from_yaml()`. Drop `mem_slurm`/`walltime_slurm` (only needed for bash script gen).

Each step is independently deployable and testable.

---

## Open questions

1. **Gateway flyweight vs injected mapper** — solve the two-gateway problem with caching (zero signature changes) or explicit injection (cleaner but more churn)?
2. **Job arrays for multi-seed** — SLURM `--array` is more efficient than N independent jobs. `make_slurm_executor` could set `slurm_array_parallelism` — handle per-backend or in `run_dag()`?
3. **Retry mechanism** — submitit raises `SlurmJobFailed` with SLURM state. Retry wrapper calls `scale_resources()` and resubmits. But in fire-and-forget mode (no coordinator alive), epilog-based retry is the only option. Support both?
4. **Dagster future** — if we ever need a web UI for monitoring, manifest-reading dashboard (already exists on HF) is cheaper than re-adding Dagster.
5. **`_preamble.sh` longevity** — submitit's `setup=` runs shell commands before the Python callable. Is the preamble the right unit, or should module loading / venv activation be separate setup steps?
