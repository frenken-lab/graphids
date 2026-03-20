# Stage Executor + Launcher: Research-First Brainstorm

**Date:** 2026-03-20
**Status:** Brainstorm (research-first)
**Prior version:** `stage-executor-and-launcher.freeform.md` (design-first, partially wrong)

---

## Problem (same as freeform version)

4 entry paths with inconsistent guarantees. `api.py` bypasses all cross-cutting concerns. Platform assumptions (SLURM) leak through 4+ files. See freeform version for the full trace.

## Goals

1. Single executor: every path gets validation, manifest, logging, archive
2. Platform-agnostic orchestration via `concurrent.futures.Executor` (stdlib)
3. Drop Dagster dependency (not using daemon mode)
4. submitit for SLURM backend

---

## Library research

### 1. `concurrent.futures.Executor` (stdlib)

**What it provides:**
- `Executor.submit(fn, *args, **kwargs) -> Future`
- `Future.result()` — blocks until done, re-raises exceptions
- `Future.done()`, `.cancel()`, `.exception()`
- `ProcessPoolExecutor` — local backend, stdlib
- `as_completed(futures)` — iterate results as they finish

**Coverage: ~90% of Launcher protocol from freeform plan.**

**Gap (10%):**
- No `depends_on` parameter — `concurrent.futures` is for independent tasks
- No resource specification — it's a generic interface
- No failure classification (OOM vs TIMEOUT)

**Gap fix:** These are SLURM-specific concerns, not executor concerns. Dependencies go through `slurm_additional_parameters`. Resources go through `update_parameters()`. Failure classification is a 10-line helper that reads `job.get_info()["State"]`.

Sources:
- [Python docs: concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html)

### 2. `submitit` (Facebook, PyPI)

**What it provides:**
- `SlurmExecutor(Executor)` — IS a `concurrent.futures.Executor`
- `AutoExecutor` — auto-detects local vs SLURM
- `executor.submit(fn, *args)` → `Job` (extends `Future`)
- `executor.map_array(fn, *iterables)` → list of Jobs (SLURM `--array`)
- `executor.update_parameters(...)`:
  - `mem_gb`, `gpus_per_node`, `cpus_per_task`, `timeout_min`, `partition`, `account`
  - `setup=["module load cuda/12.4", "source .venv/bin/activate"]` — shell commands before Python
  - `slurm_additional_parameters={"dependency": "afterok:12345", "signal": "B:USR1@180"}`
  - `slurm_array_parallelism=N` — limit concurrent array tasks
- `job.result()` — blocks, re-raises remote exceptions
- `job.job_id` — SLURM job ID string
- `job.state` — maps SLURM states to submitit states
- `job.get_info()` — full sacct-style dict (`State`, `NodeList`, etc.)
- `job.stderr()`, `job.stdout()` — log retrieval
- Checkpointing: `Checkpointable` base class with `checkpoint()` → `DelayedSubmission`
- `slurm_max_num_timeout=N` — auto-requeue on timeout up to N times

**Coverage: ~95% of SlurmLauncher + slurm_primitives.py from freeform plan.**

**Gap (5%):**
- Dependency chains (`afterok`): NOT a first-class API, but works via `slurm_additional_parameters={"dependency": "afterok:JOB_ID"}`. Confirmed in [submitit issue #23](https://github.com/facebookincubator/submitit/issues/23).
- Failure classification: `job.get_info()["State"]` returns SLURM state string. Map to our categories in ~10 lines.
- Signal handling: submitit catches SIGUSR1 for its own checkpointing. **Potential conflict with Lightning's `SLURMEnvironment(auto_requeue=True)`** which also catches SIGUSR1. See [Lightning issue #21406](https://github.com/Lightning-AI/pytorch-lightning/issues/21406). submitit recently moved away from USR1 due to NCCL conflict. **SPIKE NEEDED.**
- `ResourceSpec` → submitit kwargs mapping: ~15 lines, not a library concern.

**Key API confirmed by introspecting submitit 1.5.4:**

`SlurmExecutor` has an equivalence dict: `mem_gb`→`mem`, `timeout_min`→`time`.
Both convenience and native names work. Native `dependency` param exists (no need for `additional_parameters`).
`signal_delay_s` defaults to 90s (sends USR1 90s before SLURM timeout — built in).

```python
executor = submitit.SlurmExecutor(folder="slurm_logs")
executor.update_parameters(
    mem_gb=32,             # or mem="32G" (native)
    gpus_per_node=1,
    cpus_per_task=4,
    timeout_min=240,       # or time=240 (native)
    partition="gpu",       # native name (NOT slurm_partition)
    account="PAS1266",     # native name (NOT slurm_account)
    setup=["source scripts/slurm/_preamble.sh"],
    dependency="afterok:12345",  # native param! no additional_parameters needed
    signal_delay_s=180,    # seconds before timeout to send USR1 (default 90)
)
job = executor.submit(my_function, arg1, arg2)
print(job.job_id)       # "12345"
result = job.result()   # blocks, returns what my_function returned
```

**`AutoExecutor` vs `SlurmExecutor`:** AutoExecutor adds `slurm_` prefix (`slurm_partition`, `slurm_account`, `slurm_setup`) and convenience names (`mem_gb`, `timeout_min`). SlurmExecutor uses SLURM-native names directly plus its own equivalence dict. **Use `SlurmExecutor` directly** — clearer, no deprecation warnings, the plan only targets SLURM anyway (K8s backend would be a different executor class).

Sources:
- [submitit PyPI](https://pypi.org/project/submitit/)
- [submitit GitHub](https://github.com/facebookincubator/submitit)
- [submitit examples](https://github.com/facebookincubator/submitit/blob/main/docs/examples.md)
- [submitit checkpointing](https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md)
- [SlurmExecutor source](https://github.com/facebookincubator/submitit/blob/main/submitit/slurm/slurm.py)
- [Signal conflict: Lightning #21406](https://github.com/Lightning-AI/pytorch-lightning/issues/21406)
- [submitit signal issue #1760](https://github.com/facebookincubator/submitit/issues/1760)

### 3. `graphlib.TopologicalSorter` (stdlib, Python 3.9+)

**What it provides:**
- `TopologicalSorter(graph_dict).static_order()` → topologically sorted list
- Already used in `fire_and_forget()` today

**Coverage: 100% of dag.py from freeform plan** — topo sort is the only algorithm needed.

### 4. `submitit.AutoExecutor` — local/SLURM auto-detection

**What it provides:**
- Detects environment: if SLURM available → `SlurmExecutor`, else → `LocalExecutor`
- Same `submit()` API regardless of backend
- **This IS the "platform-agnostic launcher"** — no custom code needed

**Coverage: replaces both `SlurmLauncher` and `LocalLauncher` from freeform plan.**

**Gap:** `AutoExecutor` picks backend automatically. If you want explicit control (force local even on SLURM for testing), use `SlurmExecutor` or `LocalExecutor` directly. Both are `concurrent.futures.Executor`.

---

## What's genuinely custom (the gap after libraries)

| Need | Library | Gap | Custom code |
|------|---------|-----|-------------|
| Stage cross-cutting concerns | None — domain-specific | 100% | `execute_stage()` ~40 lines |
| ResourceSpec → submitit kwargs | submitit `update_parameters` | Mapping only | `make_executor()` ~20 lines |
| Failure classification | submitit `job.get_info()` | State string → enum | `classify_failure()` ~10 lines |
| DAG execution | `graphlib` + `concurrent.futures` | Wire together | `run_dag()` ~25 lines |
| Dependency chains | submitit native `dependency=` param | Extract `.job_id` from futures | 3 lines inside `make_slurm_executor()` |
| Multi-seed submission | submitit `map_array()` | Already built | 0 lines |
| **Total custom** | | | **~95 lines** |

---

## Spike plan

### Spike 1: submitit basic round-trip on OSC
```python
# tests/spikes/spike_submitit_basic.py
import submitit
executor = submitit.AutoExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(
    timeout_min=5, slurm_partition="cpu",
    slurm_account="PAS1266",
    setup=["source scripts/slurm/_preamble.sh"],
)
job = executor.submit(lambda x: x ** 2, 42)
print(f"Job ID: {job.job_id}, State: {job.state}")
print(f"Result: {job.result()}")  # should print 1764
```
**Validates:** preamble, pickle, result retrieval, AutoExecutor detection.

### Spike 2: PipelineConfig pickle round-trip
```python
# tests/spikes/spike_pickle_config.py
import cloudpickle, pickle
from graphids.config import resolve
cfg = resolve("vgae", "large", dataset="hcrl_sa")
cfg2 = pickle.loads(cloudpickle.dumps(cfg))
assert cfg == cfg2
print(f"Config pickle round-trip OK: {cfg2.model_type}_{cfg2.scale}")
```
**Validates:** frozen Pydantic model survives submitit's serialization.

### Spike 3: Signal handling — submitit vs Lightning SIGUSR1
**This is the highest-risk spike.** Both submitit and Lightning's `SLURMEnvironment` catch SIGUSR1. They may conflict.
```python
# tests/spikes/spike_submitit_signal.py
import submitit

class SignalTest(submitit.helpers.Checkpointable):
    def __init__(self):
        self.received = []
    def __call__(self):
        import signal, time
        signal.signal(signal.SIGUSR1, lambda s, f: self.received.append("USR1"))
        time.sleep(120)
        return self.received
    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(self)

executor = submitit.SlurmExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(
    timeout_min=2, slurm_partition="cpu", slurm_account="PAS1266",
    slurm_additional_parameters={"signal": "B:USR1@60"},
    slurm_max_num_timeout=0,  # don't requeue, just test signal
)
job = executor.submit(SignalTest())
# If submitit intercepts USR1 before our handler, received will be empty
```
**Validates:** whether we can use our own SIGUSR1 handler or need submitit's checkpointing.
**If this fails:** we may need to disable submitit's signal handling (`submitit issue #1760`) and keep our current `_preamble.sh` trap + Lightning `SLURMEnvironment`. Or use submitit's `Checkpointable` instead of Lightning's requeue.

### Spike 4: Dependency chains via additional_parameters
```python
# tests/spikes/spike_submitit_deps.py
import submitit
executor = submitit.AutoExecutor(folder="slurm_logs/spikes/%j")
executor.update_parameters(timeout_min=2, slurm_partition="cpu", slurm_account="PAS1266")

job1 = executor.submit(lambda: "first")
print(f"Job 1: {job1.job_id}")

executor.update_parameters(
    slurm_additional_parameters={"dependency": f"afterok:{job1.job_id}"}
)
job2 = executor.submit(lambda: "second")
print(f"Job 2: {job2.job_id} (depends on {job1.job_id})")
print(f"Result 1: {job1.result()}")
print(f"Result 2: {job2.result()}")
```
**Validates:** `afterok` chains work through submitit's API.

---

## What gets deleted (same as freeform version)

| File | Lines | Replaced by |
|------|------:|-------------|
| `subprocess_utils.py` | 76 | submitit submits Python callables directly |
| `slurm_primitives.py` | 245 | `submitit.SlurmExecutor` + `job.get_info()` |
| `pipes_slurm.py` | 124 | submitit (no Dagster Pipes needed) |
| `dagster_defs.py` | 271 | `graphlib` + `submitit` + `run_dag()` |
| `job.py` (partial) | ~40 | `executor.update_parameters()` takes kwargs directly |
| **Total** | **~756** | |

## What gets created

| What | Lines | Why it's custom |
|------|------:|-----------------|
| `execute_stage()` | ~40 | Domain-specific: manifest, archive, structlog context. No library does this. |
| `make_executor()` | ~20 | Maps `ResourceSpec` → `submitit.update_parameters()` kwargs. Thin adapter. |
| `run_dag()` | ~25 | Wires `graphlib.TopologicalSorter` + `submitit.submit()` + dependency passing. |
| `classify_failure()` | ~10 | Maps SLURM state strings → `FailureCategory` enum. |
| **Total** | **~95** | |

**Net: -661 lines. Custom code is only the gap between libraries and domain needs.**

---

## Migration: two commits

### Commit 1: Write new code (~160 lines)

1. **Run spikes 1-4.** Spike 3 (signal conflict) is highest risk. If any fail, update this doc before proceeding.
2. **Create `pipeline/executor.py`** (~68 lines) — `execute_stage()` + `StageResult`. Extracted from `cli.py:_run_single_stage()`.
3. **Create `pipeline/orchestration/slurm.py`** (~50 lines) — `make_slurm_executor()` + `classify_failure()`.
4. **Create `pipeline/orchestration/dag.py`** (~55 lines) — `DagNode` (moved from dagster_defs), `build_dag_topology` (moved from dagster_defs), `run_dag()`, `get_resources` + `RESOURCE_PROFILES` + `FAILURE_REACTIONS` + `scale_resources` + `SlurmJobFailed` (moved from slurm_primitives).
5. **Wire entry points** — `cli.py:_run_single_stage()` → calls `execute_stage()`. `api.py:train()` → calls `execute_stage()`. `cli.py:orchestrate` → calls `run_dag(make_slurm_executor, ...)`.
6. **At this point:** old files still exist but nothing imports from them. Both code paths work.

### Commit 2: Delete old code (~756 lines, pure deletion)

Checklist — every item is a deletion, no new code in this commit:

**Delete entire files:**
- [ ] `graphids/pipeline/subprocess_utils.py` (76 lines) — `build_cli_cmd()`
- [ ] `graphids/pipeline/orchestration/pipes_slurm.py` (124 lines) — `PipesSlurmClient`, `submit_no_poll`
- [ ] `graphids/pipeline/orchestration/dagster_defs.py` (271 lines) — all Dagster assets, `fire_and_forget`, `defs`
- [ ] `graphids/pipeline/orchestration/slurm_primitives.py` (245 lines) — `generate_sbatch_script`, `submit_sbatch`, `poll_until_done`, `sacct_query`, `write_script_file`

**Delete from `job.py`:**
- [ ] `ResourceSpec.from_yaml()` (lines 44-82, ~39 lines) — replace with Pydantic `field_validator`
- [ ] `ResourceSpec.mem_slurm` property (lines 30-32) — only used by deleted `generate_sbatch_script`
- [ ] `ResourceSpec.walltime_slurm` property (lines 34-42) — only used by deleted `generate_sbatch_script`

**Delete from `cli.py`:**
- [ ] `_run_single_stage()` function — replaced by `execute_stage()`
- [ ] `_init_pipes_context()` function — Dagster Pipes removed

**Delete from `pipeline/__init__.py`:**
- [ ] `from graphids.pipeline.subprocess_utils import build_cli_cmd` — file deleted

**Delete from `pipeline/orchestration/__init__.py`:**
- [ ] All re-exports from deleted modules

**Remove dependencies from `pyproject.toml`:**
- [ ] `dagster`
- [ ] `dagster-pipes`

**Verify:** `grep -r "dagster\|pipes_slurm\|subprocess_utils\|slurm_primitives\|build_cli_cmd\|generate_sbatch\|PipesSlurmClient\|submit_no_poll\|fire_and_forget" graphids/` returns nothing.

### Line count

| | Lines |
|---|---:|
| Commit 1: new code written | ~160 |
| Commit 1: code moved (DagNode, build_dag_topology, get_resources, etc.) | ~85 |
| Commit 2: code deleted | ~670 |
| **Net change** | **-596** |

## Alternatives considered and deferred

### Parsl + RadicalPilotExecutor (heterogeneous pilot jobs)
RADICAL-Pilot uses a **pilot-job model**: one big SLURM allocation, then an internal scheduler dispatches tasks with different CPU/GPU/memory to slices of that allocation. Per-task heterogeneous resources, automatic DAG from futures, SLURM + K8s portability.

**Why deferred:** Pilot model optimizes for many short tasks (100s-1000s) where queue wait dominates. Our pipeline has ~30 long tasks (1-4h each) where compute dominates. One queue wait per stage (job-per-task) vs one queue wait total (pilot) saves ~45 min on a 12-hour pipeline — not enough to justify RADICAL-Pilot's complexity (needs MongoDB, heavy runtime). Revisit if task count grows significantly (HPO with 100s of trials).

### Parsl + FluxExecutor (nested scheduler)
Flux is a mini-SLURM that runs *inside* a SLURM allocation. Each task becomes a Flux sub-job with its own resource spec. True job-per-task with per-task resources.

**Why deferred:** Requires Flux installed on OSC (not verified). Adds a nested scheduler layer. Our `submitit + graphlib` approach is simpler and already job-per-task. Revisit if OSC adds Flux or we move to a Flux-native system (LLNL, etc).

### Parsl + multi-executor (different profiles per executor)
Multiple HighThroughputExecutors with different resource profiles, route tasks to the right one. Gets heterogeneous resources without Flux.

**Why deferred:** Still uses worker pool model — each executor holds a SLURM allocation for its full lifetime. On a capped PAS1266 allocation, idle GPU blocks waste SUs. Job-per-task (submitit) allocates exactly what each stage needs and releases after.

## Signal conflict: submitit vs Lightning SIGUSR1

### The problem
Both systems catch SIGUSR1 for checkpoint-and-requeue on SLURM timeout:

- **submitit**: catches SIGUSR1 → calls `callable.checkpoint()` → pickles state → resubmits job via `DelayedSubmission`. Expects the callable to implement `Checkpointable`.
- **Lightning `SLURMEnvironment(auto_requeue=True)`**: catches SIGUSR1 → saves `.pl_auto_save.ckpt` → calls `scontrol requeue $SLURM_JOB_ID`. Expects to own the signal handler.
- **Our `_preamble.sh`**: catches SIGUSR1 in bash → forwards to child Python process via `kill -USR1`. A third participant.

Only one Python signal handler can be registered for SIGUSR1. Whoever registers last wins.

### What the research found
- [Lightning issue #21406](https://github.com/Lightning-AI/pytorch-lightning/issues/21406): signal handlers in checkpointing lead to intermittent failures when they run during backward pass.
- [submitit issue #1760](https://github.com/facebookincubator/submitit/issues/1760): users requesting ability to turn off submitit's signal handling.
- submitit recently moved away from USR1 due to NCCL conflict (NCCL also uses USR1 internally).

### Options (spike 3 will determine which)

| Option | How | Pros | Cons |
|---|---|---|---|
| **A. Disable submitit signal handling** | Set `slurm_max_num_timeout=0` + don't implement `Checkpointable` | Lightning keeps auto-requeue, no change to training code | Lose submitit's checkpoint/resume for non-Lightning tasks |
| **B. Use submitit checkpointing, drop Lightning's** | Don't use `SLURMEnvironment(auto_requeue=True)`, implement `Checkpointable` on `execute_stage` | One checkpoint system, submitit manages requeue | Must rewrite checkpoint logic, Lightning's is battle-tested |
| **C. Keep bash trap, bypass both** | `_preamble.sh` trap catches USR1, forwards to Python. Disable both submitit and Lightning signal handlers. Lightning saves via its normal callback. | Proven pattern (current system works) | Three layers (bash → Python → Lightning), fragile chain |
| **D. Use a different signal** | Configure `#SBATCH --signal=B:USR2@180`, register Lightning on USR2 | No conflict — each system gets its own signal | Lightning hardcodes USR1 in `SLURMEnvironment` — may need subclass or patch |

**Recommendation:** Start with **Option A** (simplest). submitit just submits and waits. Lightning handles its own checkpoint/requeue as it does today. submitit doesn't need to checkpoint because `execute_stage` is the callable — it just resolves config and calls a stage function. There's nothing to resume at the executor level; Lightning handles mid-training resumption.

## Open questions

1. **Signal conflict (spike 3):** Option A (disable submitit signals) is the plan. Spike validates it works.
2. **`AutoExecutor` vs explicit:** `AutoExecutor` auto-detects SLURM. But tests on login nodes would auto-detect SLURM and try to submit. Use explicit `LocalExecutor` for tests, `AutoExecutor` or `SlurmExecutor` for real runs.
3. **Gateway flyweight:** `open_gateway(cfg)` called in both executor and stage functions. Add `@lru_cache` keyed on config identity, or pass mapper as arg?
4. **Job arrays:** submitit `map_array()` handles multi-seed natively. Should `run_dag()` use it for seed parallelism, or submit individual jobs with dependencies?
5. **`build_dag_topology()` coupling:** Currently calls `resolve("vgae", "large")` to read `cfg.variants`. DAG definition shouldn't depend on resolving a specific model config. Should read from `pipeline.yaml` directly. Low priority — works now, fragile later.

---

## Draft implementation

### `pipeline/executor.py`

```python
"""Stage executor: single entry point for ALL stage execution.

Every path through the pipeline (CLI, API, submitit, notebook) calls
execute_stage(). Cross-cutting concerns live here, nowhere else.
"""
from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import structlog

from graphids.config import PipelineConfig
from graphids.logging import configure_logging
from graphids.pipeline.validate import validate
from graphids.storage import open_gateway, write_manifest

log = structlog.get_logger()


@dataclass(frozen=True)
class StageResult:
    metrics: dict[str, float]
    duration_seconds: float
    checkpoint_path: Path | None
    manifest_path: Path


def execute_stage(cfg: PipelineConfig, stage: str) -> StageResult:
    """Execute a pipeline stage with full guarantees.

    Owns: logging setup, validation, structlog context, archive/restore,
    config snapshot, timing, manifest write. Stage functions do the ML work.
    """
    configure_logging()  # idempotent
    structlog.contextvars.bind_contextvars(
        dataset=cfg.dataset, model=cfg.model_type, scale=cfg.scale,
        stage=stage, seed=cfg.seed,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
    )
    validate(cfg, stage)

    gw, mapper = open_gateway(cfg)
    sdir = gw.resolve(stage)

    # Archive previous run (restore on failure)
    archive = None
    if (sdir / "config.json").exists():
        archive = sdir.parent / f"{sdir.name}.archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sdir.rename(archive)
        log.warning("run_archived", path=str(archive))

    mapper.save_config(cfg, stage)
    log.info("run_started")
    t0 = time.monotonic()

    try:
        from graphids.pipeline import STAGE_FNS

        result = STAGE_FNS[stage](cfg)
        duration = time.monotonic() - t0

        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        metrics["duration_seconds"] = duration

        manifest_path = sdir / "_manifest.json"
        try:
            write_manifest(
                sdir, dataset=cfg.dataset, model_type=cfg.model_type,
                scale=cfg.scale, stage=stage,
                auxiliaries=cfg.auxiliaries[0].type if cfg.auxiliaries else "none",
                seed=cfg.seed, metrics=metrics,
            )
        except Exception as e:
            log.warning("manifest_write_failed", error=str(e))

        if archive and archive.exists():
            shutil.rmtree(archive, ignore_errors=True)

        ckpt = sdir / "best_model.pt"
        log.info("stage_complete", **{k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        return StageResult(metrics, duration, ckpt if ckpt.exists() else None, manifest_path)

    except Exception:
        if archive and archive.exists():
            if sdir.exists():
                shutil.rmtree(sdir, ignore_errors=True)
            archive.rename(sdir)
        raise
```

### `pipeline/orchestration/slurm.py`

```python
"""Thin adapter: ResourceSpec -> submitit.SlurmExecutor.

The only SLURM-specific code in the project. Everything else uses
concurrent.futures.Executor (which submitit.SlurmExecutor implements).
"""
from __future__ import annotations

from enum import Enum

import submitit

from graphids.config import SLURM_ACCOUNT
from .job import ResourceSpec


class FailureCategory(Enum):
    OOM = "oom"
    TIMEOUT = "timeout"
    INFRA = "infra"
    APPLICATION = "application"


_SLURM_FAILURE_MAP = {
    "OUT_OF_MEMORY": FailureCategory.OOM,
    "TIMEOUT": FailureCategory.TIMEOUT,
    "NODE_FAIL": FailureCategory.INFRA,
    "PREEMPTED": FailureCategory.INFRA,
}


def classify_failure(job: submitit.Job) -> FailureCategory:
    """Map SLURM job state to a platform-agnostic failure category."""
    state = job.get_info().get("State", "FAILED").split()[0]
    return _SLURM_FAILURE_MAP.get(state, FailureCategory.APPLICATION)


def make_slurm_executor(
    resources: ResourceSpec,
    dep_futures: list | None = None,
    *,
    setup: list[str] | None = None,
    log_folder: str = "slurm_logs/%j",
) -> submitit.SlurmExecutor:
    """Configure a submitit SlurmExecutor from a ResourceSpec.

    dep_futures: list of submitit.Job (or any Future with .job_id).
    SLURM-specific: extracts job IDs and passes --dependency=afterok.
    """
    executor = submitit.SlurmExecutor(folder=log_folder)

    dep_str = None
    if dep_futures:
        dep_ids = [str(f.job_id) for f in dep_futures]
        dep_str = f"afterok:{':'.join(dep_ids)}"

    executor.update_parameters(
        mem_gb=resources.memory_gb,          # equivalence: mem_gb -> mem
        gpus_per_node=resources.gpus,
        cpus_per_task=resources.cpus,
        timeout_min=int(resources.walltime.total_seconds() // 60),  # equivalence: timeout_min -> time
        partition=resources.partition,        # native SLURM name
        account=SLURM_ACCOUNT,               # native SLURM name
        setup=setup or ["source scripts/slurm/_preamble.sh"],
        dependency=dep_str,                  # native param (not additional_parameters)
        exclude=resources.exclude_nodes or None,
        signal_delay_s=180,                  # send USR1 180s before timeout
    )
    return executor
```

### `pipeline/orchestration/__init__.py` (extend with `run_dag`)

```python
"""Pipeline orchestration: DAG execution via concurrent.futures.

run_dag() is platform-agnostic. It takes a factory that returns a
concurrent.futures.Executor — submitit.SlurmExecutor for HPC,
ProcessPoolExecutor for local, anything that implements .submit().
"""
from __future__ import annotations

import graphlib
from concurrent.futures import Future
from typing import Callable

import structlog

from graphids.config import resolve
from graphids.pipeline.executor import execute_stage
from .dagster_defs import DagNode, build_dag_topology  # reuse topology builder
from .slurm_primitives import get_resources, scale_resources, FAILURE_REACTIONS

log = structlog.get_logger()

# Re-export existing public API
from .job import ResourceSpec
from .slurm_primitives import SlurmJobFailed


def run_dag(
    executor_factory: Callable[[ResourceSpec, list[Future]], object],
    dag: dict[str, DagNode],
    dataset: str,
    seeds: list[int],
    *,
    dry_run: bool = False,
) -> dict[str, Future]:
    """Execute pipeline DAG through any concurrent.futures.Executor.

    Parameters
    ----------
    executor_factory
        Callable(resources, dep_futures) -> Executor with .submit().
        Each backend decides how to handle dependencies:
        - SLURM: extract .job_id from futures, pass as afterok
        - Local: call .result() to block before submitting
        - K8s: watch Job completion, submit next
    """
    topo_order = list(graphlib.TopologicalSorter(
        {name: set(node.deps) for name, node in dag.items()}
    ).static_order())

    all_futures: dict[str, Future] = {}
    for seed in seeds:
        futures: dict[str, Future] = {}
        for name in topo_order:
            node = dag[name]
            dep_futures = [futures[d] for d in node.deps if d in futures]
            resources = get_resources(node.resource_model, node.scale, node.stage)
            cfg = resolve(node.cli_model, node.scale, dataset=dataset, seed=seed)

            if dry_run:
                log.info("dry_run", asset=name, deps=[str(d) for d in node.deps])
                continue

            executor = executor_factory(resources, dep_futures)
            futures[name] = executor.submit(execute_stage, cfg, node.stage)
            all_futures[f"{name}__seed{seed}"] = futures[name]

    return all_futures
```

### Usage: wiring it together

```python
# In cli.py orchestrate command — replaces fire_and_forget()
from graphids.pipeline.orchestration import run_dag, build_dag_topology
from graphids.pipeline.orchestration.slurm import make_slurm_executor

dag = build_dag_topology()
futures = run_dag(
    executor_factory=lambda r, deps: make_slurm_executor(r, dep_futures=deps),
    dag=dag, dataset="hcrl_sa", seeds=[42, 123, 456],
)

# Local mode (tests, notebooks) — block on deps, run in subprocess:
from concurrent.futures import ProcessPoolExecutor
def local_executor(resources, dep_futures):
    for f in dep_futures:
        f.result()  # block until dep finishes
    return ProcessPoolExecutor(1)

futures = run_dag(
    executor_factory=local_executor,
    dag=dag, dataset="hcrl_sa", seeds=[42],
)

# api.py — single stage, full guarantees:
from graphids.pipeline.executor import execute_stage
result = execute_stage(cfg, "autoencoder")
```
