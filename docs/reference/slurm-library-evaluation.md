# HPC / Workflow Library Evaluation

Evaluated 2026-04-09. Context: GraphIDS uses a 3-stage KD pipeline
(VGAE→GAT→fusion) orchestrated by Monarch actors in a single SLURM
allocation on OSC Pitzer. 940 lines of custom SLURM code in
`graphids/slurm/`, Typer CLI, jsonnet configs. This evaluates all
libraries from `docs/reference/hpc-libraries.md`.

## Environment Constraints (OSC Pitzer)

| Fact | Value |
|------|-------|
| SLURM version | **25.05.4** |
| `libslurm.so` | `/usr/lib64/libslurm.so.43.0.0` (present) |
| `libslurmfull.so` | `/usr/lib64/slurm/libslurmfull.so` (10MB, in plugin dir) |
| SLURM headers | `/usr/include/slurm/` (slurm-devel installed) |
| Python | 3.12.4 via `module load python/3.12` |

---

## 1. PySlurm (`PySlurm/pyslurm`)

**One-liner:** Cython bindings to SLURM's C API (`libslurmfull.so`), providing
typed Python objects for jobs, nodes, partitions, and the accounting database.

### Architecture

Cython (86% of codebase) compiled against SLURM's internal shared library.
Not a subprocess wrapper — direct C-level function calls. This gives
structured return types instead of parsed text, but creates a hard build-time
and runtime dependency on the exact SLURM shared library.

### Installation

Requires at build time:
- SLURM shared library (`libslurmfull.so`) + headers
- Cython >= 0.29.37
- Python >= 3.6

```bash
export SLURM_INCLUDE_DIR=/usr/include/slurm
export SLURM_LIB_DIR=/usr/lib64
pip install pyslurm  # or build from source with scripts/build.sh
```

### API Surface

Rich typed API across multiple domains:

| Class | Purpose | Equivalent CLI |
|-------|---------|----------------|
| `JobSubmitDescription` | Construct + submit jobs | `sbatch` |
| `Job` / `Jobs` | Query, cancel, hold, modify running jobs | `scontrol`, `scancel` |
| `Job.load_stats()` | Real-time job performance metrics | `sstat` |
| `db.Job` / `db.JobFilter` | Historical accounting queries | `sacct` |
| `db.JobStatistics` | Aggregated job statistics | `sacct` aggregations |
| `Partition` / `Node` | Cluster topology queries | `sinfo`, `scontrol` |
| `Reservation` | Reservation management | `scontrol` |

**Job submission example:**
```python
import pyslurm

desc = pyslurm.JobSubmitDescription(
    name="autoencoder_hcrl_sa",
    cpus_per_task=8,
    memory_per_node="40G",
    time_limit="4:00:00",
    partitions="gpu",
    gpus_per_node="1",
    script="/path/to/run.sh",
)
job_id = desc.submit()  # returns int

# Later: query
job = pyslurm.Job.load(job_id)
print(job.state, job.elapsed_time)

# Cancel
job.cancel()
```

**Accounting (sacct replacement):**
```python
from pyslurm import db

conn = db.Connection()
job_filter = db.JobFilter(users=["rf15"], start_time="2026-04-01")
jobs = db.Jobs.load(conn, job_filter)
for j in jobs:
    print(j.id, j.elapsed_time, j.stats.max_rss)
```

### SLURM Version Coupling

**Strict major.minor match required.** PySlurm X.Y.Z works only with SLURM X.Y.*.

Available PySlurm releases on PyPI:

| PySlurm version | Target SLURM | Release date |
|-----------------|--------------|--------------|
| 25.11.1 | 25.11.x | 2026-04-01 |
| 25.5.0 | 25.05.x | 2025-11-04 |
| 24.11.0 | 24.11.x | 2024-12-30 |
| 24.5.x | 24.05.x | ~2024 |
| 23.11.x | 23.11.x | 2024-12-28 |
| 18.8.1.1 | 18.8.x | 2018-10-11 |

Recent releases track SLURM versions closely. PySlurm 25.5.0 exists for
OSC's SLURM 25.05.4, but `libslurmfull.so` is still the hard blocker.
Note the **7-year PyPI gap** between 18.8 (2018) and 23.11 (2024) — intermediate
versions were only available by building from GitHub branches.

### Maintenance

- **License:** GPL-2.0-only
- **Stars:** ~500
- **Bus factor:** Effectively 1 (Toni Harzendorf for the modern rewrite)
- **Release cadence:** Sporadic — one release per SLURM major, but the
  PyPI gap (2018 → 2026) suggests development happened on GitHub branches
  without PyPI releases for years

### OSC Compatibility

~~Previously reported as blocked due to missing `libslurmfull.so`.~~
**Corrected:** `libslurmfull.so` exists at `/usr/lib64/slurm/libslurmfull.so`
(shipped by the base `slurm` RPM). It was missed because it lives in the
plugin subdirectory, not the standard lib path. PySlurm's `setup.py`
searches the `slurm/` subdirectory, so it should find it.

**Remaining concerns:**

1. **SLURM version match.** OSC runs 25.05.4; PySlurm 25.5.0 exists and
   should work. However, it's a single release with no patch updates.

2. **Build complexity.** Cython compilation on OSC requires `module load`
   for the right Python and a working C compiler toolchain. Any SLURM
   upgrade on OSC requires a PySlurm rebuild.

3. **`libslurmfull.so` has no ABI stability guarantees.** SchedMD
   considers it an internal library (bug #4449, resolved WONTFIX). Internal
   functions change at micro release granularity — a SLURM point release
   (e.g., 25.05.4 → 25.05.5) could break PySlurm without warning.

4. **GPL-2.0 license** — copyleft, relevant if GraphIDS is ever distributed.

### Verdict

**Technically viable on OSC but marginal benefit.** `libslurmfull.so` is
present, PySlurm 25.5.0 matches SLURM 25.05. The real question is whether
replacing ~330 lines of subprocess sacct parsing with PySlurm's `db.*`
classes justifies the Cython build dependency, version-coupling maintenance,
and GPL license. For the current scale of SLURM interaction (5 subprocess
calls), the answer is no.

---

## 2. simple_slurm (`amq92/simple_slurm`)

**One-liner:** Pure Python subprocess wrapper that generates `#SBATCH` scripts
programmatically and submits them via the system's `sbatch`/`srun` commands.

### Architecture

Zero native code. Constructs shell script strings with `#SBATCH` directives,
then calls `subprocess.run(["sbatch", ...])`. The generated script is
inspectable before submission (just `print(slurm)`). Also wraps `squeue`
and `scancel` for basic job monitoring and cancellation.

### Installation

```bash
pip install simple_slurm     # PyPI
conda install -c conda-forge simple_slurm  # conda
```

- **No build dependencies** — pure Python, no SLURM headers or libs needed
- **No SLURM version coupling** — works with any SLURM that has `sbatch` on `$PATH`
- **Python version:** Not explicitly pinned; pure Python, works with 3.12+

### API Surface

| Feature | Method | Notes |
|---------|--------|-------|
| Configure job | `Slurm(partition="gpu", time="4:00:00", ...)` | All sbatch flags as kwargs |
| Add shell commands | `slurm.add_cmd("module load python")` | Prepended to script |
| Submit (detached) | `slurm.sbatch("python train.py")` | Returns job ID |
| Submit (blocking) | `slurm.srun("python train.py")` | Waits for completion |
| Query queue | `slurm.squeue.display_jobs()` | Wraps `squeue` |
| Cancel job | `slurm.scancel.cancel_job(job_id)` | Wraps `scancel` |
| Inspect script | `print(slurm)` | Shows generated `#SBATCH` script |

**No sacct/accounting support.** Query and monitoring are limited to
`squeue` (running jobs only) and `scancel`.

**Flexible argument syntax (multiple equivalent forms):**
```python
from simple_slurm import Slurm

# All equivalent:
slurm = Slurm(array="3-11")
slurm = Slurm(array=range(3, 12))
slurm = Slurm("--array", "3-11")
slurm.set_array(range(3, 12))

# Dependencies as dict, list, or string:
slurm.set_dependency(dict(after=65541, afterok=34987))

# datetime.timedelta for time:
slurm = Slurm(time=datetime.timedelta(hours=4))
```

**Full submission example:**
```python
from simple_slurm import Slurm

slurm = Slurm(
    job_name="autoencoder_hcrl_sa",
    partition="gpu",
    gres="gpu:1",
    cpus_per_task=8,
    mem="40G",
    time="4:00:00",
    output="slurm_logs/%j.out",
    signal="B:USR1@300",
)
slurm.add_cmd("source scripts/slurm/_preamble.sh")
job_id = slurm.sbatch("python -m graphids fit --config configs/stages/autoencoder.jsonnet")
```

### SLURM Version Coupling

**None.** Generates text scripts and calls CLI tools. Works with any SLURM
version that supports the requested `#SBATCH` flags. Cluster upgrades are
transparent.

### Maintenance

- **License:** AGPL-3.0 (copyleft — viral licensing)
- **Stars:** ~200
- **Open issues:** 5 (as of 2026-04-09)
- **Bus factor:** 1 (amq92)
- **Release cadence:** Periodic; available on both PyPI and conda-forge
- **Codebase size:** Small (~500 lines), easy to vendor if abandoned

### Limitations

1. **No sacct/accounting support.** Cannot query completed job history,
   wall time, RSS, or CPU efficiency. Our `graphids/slurm/core/accounting.py`
   and `ops/profile.py` (330 lines) have no replacement here.

2. **No structured return types.** Job submission returns a job ID string;
   queue queries return text. No Pydantic models or typed objects.

3. **Submission only.** The value proposition is script generation + sbatch
   call. For projects that already have a working `submit.sh`, the overlap
   is significant.

4. **`logging.basicConfig()` called at import time** (issue #42).
   `scancel.py` calls `logging.basicConfig()` at module level, hijacking
   the root logger. Conflicts with `graphids.log` structured logging.

5. **`squeue` wrapper crashes on array jobs** (issue #44).
   `update_squeue()` raises `ValueError` when parsing array job IDs like
   `57829110_[5-100]`. GraphIDS uses job arrays, so this is a blocker
   for the squeue feature.

6. **Open issue #49:** Some sbatch arguments require `--key=value` syntax
   (with `=`) but the library generates `--key value` (with space).

7. **Backslash/backtick escaping issues** (issue #45). The `script()`
   method mishandles `\` and backticks when `convert=True`.

### Verdict

**Viable but marginal benefit.** Installs trivially, no version coupling,
clean API for script generation. However, GraphIDS already has `submit.sh`
for job submission and needs sacct parsing (which simple_slurm doesn't
provide). The main value would be replacing shell script generation with
Python, but `submit.sh` already works and is well-integrated.

---

## Comparison Matrix

| Criterion | PySlurm | simple_slurm | Current (`graphids/slurm/`) |
|-----------|---------|--------------|----------------------------|
| Architecture | Cython C bindings | subprocess wrapper | subprocess wrapper |
| Install on OSC | Viable (Cython build + version pin) | `pip install` | N/A (already here) |
| SLURM version coupling | Strict major.minor | None | None |
| Job submission | `JobSubmitDescription.submit()` | `Slurm.sbatch()` | `submit.sh` (shell) |
| sacct/accounting | `db.Job`, `db.JobFilter` | **Not supported** | `sacct_query()`, `job_accounting()` |
| Job monitoring | `Job.load()`, `Jobs.load()` | `squeue` wrapper | Not in Python (squeue in shell) |
| Resource profiles | Not applicable | Not applicable | `ResourceSpec`, `get_resources()` |
| Data staging | Not applicable | Not applicable | `stage_data()` |
| Typed returns | Cython objects | Strings/ints | Pydantic `ResourceSpec` |
| Lines of code | ~15k (Cython) | ~500 (Python) | 940 (Python) |
| License | GPL-2.0 | AGPL-3.0 | N/A |

## What Would Each Replace?

### PySlurm (if it worked on OSC)

Could replace:
- `core/accounting.py` (85 lines) — `sacct_query`, `job_accounting` → `db.Job`
- `ops/profile.py` (245 lines) — sacct aggregation → `db.JobStatistics`
- Parts of `ops/staging.py` — N/A (staging is project-specific)
- `resources.py` resource lookup — N/A (project-specific profile logic)

Would NOT replace:
- `ResourceSpec` model + profile lookup (355 lines) — project-specific
- `ops/staging.py` (180 lines) — project-specific data staging
- `env.py` (39 lines) — project-specific env var reads

**Net savings if viable:** ~300 lines, plus typed accounting objects instead
of parsed text. But not viable due to OSC constraints.

### simple_slurm

Could replace:
- Parts of `submit.sh` — sbatch script generation moves to Python
- Job cancellation (currently manual) — `scancel` wrapper

Would NOT replace:
- `core/accounting.py` — no sacct support
- `ops/profile.py` — no sacct support
- `resources.py` — no resource profile concept
- `ops/staging.py` — project-specific
- `env.py` — project-specific

**Net savings:** Near zero. The submission logic lives in shell already
and simple_slurm doesn't cover accounting, which is the bulk of the
Python SLURM code.

## Recommendation

**Neither library justifies adoption at this time.**

- **PySlurm** is architecturally the right tool (typed C-level access to the
  full SLURM API) but is blocked by OSC's missing `libslurmfull.so` and
  SLURM version mismatch. Even if unblocked, the GPL-2.0 license and tight
  version coupling create ongoing maintenance burden.

- **simple_slurm** installs trivially but solves a problem GraphIDS doesn't
  have (script generation) while missing the problem it does have (sacct
  parsing). At ~500 lines it's small enough to vendor, but there's nothing
  to vendor for.

The current `graphids/slurm/` module (940 lines) is well-scoped: `ResourceSpec`
handles resource profiles, `accounting.py` parses sacct, `profile.py`
aggregates job stats, and `staging.py` manages the data tier. None of these
are generic SLURM wrappers — they encode project-specific logic (dataset
scaling, identity hash paths, 3-tier staging) that no library can replace.

**If OSC later installs `libslurmfull.so`**, the accounting layer
(`core/accounting.py` + `ops/profile.py`, ~330 lines) would be a good
candidate for replacement with PySlurm's `db.*` classes. Monitor PySlurm
releases against OSC SLURM versions.

**Other library worth noting:** Meta's **submitit** (`facebookincubator/submitit`)
supports SLURM natively with Python function pickling, auto-requeuing, and
job management. Used by FAIR for large-scale ML training. Different scope
(it replaces the submission workflow, not accounting), but relevant if
GraphIDS ever moves to programmatic job submission from Python.

---
---

# Additional HPC Libraries

Evaluated from `docs/reference/hpc-libraries.md`. Grouped by category.

---

## 3. pyslurmutils (`esrf/pyslurmutils`)

**One-liner:** `concurrent.futures.Executor` for SLURM via the SLURM REST
API (`slurmrestd`).

| Aspect | Detail |
|--------|--------|
| Architecture | REST API client — HTTP + JWT auth to `slurmrestd` daemon |
| Install | `pip install pyslurmutils` (MIT). Deps: requests, pydantic>=2, pyjwt |
| Python | >=3.8 |
| SLURM coupling | None (REST API), but **requires `slurmrestd` daemon running** |
| Maintenance | v1.5.0, last activity 2026-03-30. 0 stars (GitLab, ESRF internal). Bus factor: 1 team |

**API:** `SlurmRestExecutor(url, user, token)` — standard `executor.submit(fn, *args)`
returning `Future`s. Also `SlurmScriptRestClient` for raw script submission.

**BLOCKER: `slurmrestd` is not available on OSC Pitzer.** Verified: command
not found, no REST endpoint exposed. Library is unusable without admin
intervention. Even if available, the executor model (serialize Python callables)
doesn't match GraphIDS's sbatch-script workflow.

**Verdict: Not viable on OSC.**

---

## 4. slurm-pipeline (`acorg/slurm-pipeline`)

**One-liner:** Python CLI for multi-step shell-script pipelines with SLURM
dependency tracking, plus a thin `SAcct` class.

| Aspect | Detail |
|--------|--------|
| Architecture | Subprocess wrapper — calls `sbatch`/`sacct` via `subprocess.check_output` |
| Install | `pip install slurm-pipeline`. Deps: **pandas, plotly**, toml |
| Python | 3.6–3.9 tested (no 3.12 CI) |
| SLURM coupling | None |
| Maintenance | v4.1.2 (2024-10-26). 60 stars. Bus factor: 1. No activity since Oct 2024 |

**Pipeline model:** TOML spec declares steps with shell scripts. Each script
calls `sbatch` and prints `TASK: <name> <job_id>` to stdout. The library
chains steps via `--dependency=afterok:JOBID`. Does not match GraphIDS's
jsonnet→instantiate→sbatch flow.

**SAcct class:** ~90 lines, queries `sacct -P --format ... --jobs`. Functionally
equivalent to `graphids/slurm/core/accounting.py` (85 lines) but adds a
pandas dependency and raises on missing job IDs (vs. graceful handling).

**Verdict: Not worth adopting.** Heavy deps (pandas+plotly) for plotting
features, pipeline model doesn't fit, SAcct replaces 85 lines with 90 lines
plus a pandas dependency. Stale maintenance.

---

## 5. Parsl (`parsl-project/parsl`)

**One-liner:** Python-native parallel workflow library with built-in SLURM
provider and pilot-job auto-scaling.

| Aspect | Detail |
|--------|--------|
| Architecture | Data Flow Kernel (DFK) on login node + `HighThroughputExecutor` pilot jobs via ZMQ |
| Install | `pip install parsl` (Apache-2.0). Deps: pyzmq, dill, typeguard, psutil |
| Python | >=3.10 |
| SLURM coupling | None — uses `SlurmProvider` (sbatch subprocess) |
| Maintenance | **Weekly calver releases** (2026.4.6). 610 stars. UChicago/Argonne. Bus factor: 3–4 |

**How it works:** Decorate functions with `@python_app`/`@bash_app`. Parsl
resolves the DAG via `AppFuture` data dependencies. `SlurmProvider` submits
pilot SLURM jobs ("blocks") that run worker agents. Workers connect back to
the DFK over ZMQ. Auto-scales blocks based on pending task count.

```python
from parsl import Config, python_app
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor

@python_app
def train_stage(config: dict) -> dict:
    from graphids.core.train_entrypoint import run_training
    return run_training(config)

# DAG via futures
vgae = train_stage(vgae_config)
gat = train_stage(gat_config)
fusion = train_stage({**fusion_cfg, "vgae_ckpt": vgae, "gat_ckpt": gat})
```

**vs. Monarch actors:** Monarch runs actors within a single SLURM allocation
with shared-memory IPC and GPU-aware placement — ideal for the tightly-coupled
VGAE→GAT→fusion pipeline. Parsl runs isolated processes across potentially
multiple SLURM jobs with serialized IPC. **Monarch is better for the current
pipeline; Parsl is better for embarrassingly-parallel sweeps across many jobs.**

**Limitations:** DFK must stay alive (SSH dies → work lost). Large objects
must be file paths, not serialized. Worker startup cost 5–10s (torch+PyG import).
`exclusive=True` by default wastes allocation. NFS state writes can be slow.

**Verdict: Not needed now, strongest candidate for future sweeps.** If
GraphIDS later runs large hyperparameter sweeps (50+ configs in parallel),
Parsl's auto-scaling pilot-job model would be significantly cleaner than
managing sbatch arrays. Complements (not replaces) Monarch.

---

## 6. Globus Compute (formerly funcX)

**One-liner:** Federated Function-as-a-Service — register Python functions
with a Globus cloud service, execute them on user-managed HPC endpoints.

| Aspect | Detail |
|--------|--------|
| Architecture | Client → Globus cloud service → endpoint daemon on login node → SLURM workers |
| Install | `pip install globus-compute-sdk` (Apache-2.0). Endpoint: separate package |
| Python | >=3.10 |
| SLURM coupling | None — uses same `SlurmProvider` as Parsl |
| Maintenance | SDK v4.9.0. 161 stars. UChicago/Globus. Bus factor: 2–3 |

**Key constraint:** All task routing goes through `compute.api.globus.org`.
Requires outbound HTTPS + Globus OAuth2 authentication. 10 MB payload limit.
20 req/10s rate limit. Task TTL 2 weeks. Results expire 30 min after completion.

**vs. raw sbatch:** Adds remote submission (laptop → OSC without SSH) and
fire-and-forget durability. But adds cloud dependency, auth complexity, and
payload limits. For local-only use (already on OSC), strictly more complex.

**vs. Monarch:** Fundamentally different. Monarch = shared-memory GPU actors
in one allocation. Globus Compute = isolated functions routed through cloud.
For co-located stages, Monarch is categorically better.

**Verdict: Not relevant.** Solves cross-site federation (submit from anywhere
to any endpoint). GraphIDS runs entirely on OSC with direct SLURM access.
Would only matter if triggering training from outside OSC (CI, notebooks).

---

## 7. Garden AI (`garden-ai/garden`)

**One-liner:** FAIR framework for publishing, discovering, and remotely
executing pre-trained ML models via Globus Compute.

| Aspect | Detail |
|--------|--------|
| Architecture | CLI publishes containerized models to web catalog; inference dispatched via Globus Compute |
| Install | `pipx install garden-ai` (MIT). Requires Globus account |
| Maintenance | v3.2.3 (2026-03-18). 39 stars. NSF-funded |

**Verdict: Not relevant.** Solves model *publishing and remote inference*,
not model *training*. Would only matter after GraphIDS produces finished
models to share. No overlap with training pipeline needs.

---

## 8. Globus Cascade

**One-liner:** Framework for concurrently training and using ML surrogate
models during atomistic (molecular dynamics) simulations.

| Aspect | Detail |
|--------|--------|
| Architecture | Domain-specific (ASE, CP2K, MACE). 97.6% Jupyter Notebook |
| Install | `conda env create --file environment.yml`. No PyPI package |
| Maintenance | 87 commits, 2 stars, last commit 2024-12-06. Research prototype |

**Verdict: Not relevant.** Hardwired to molecular dynamics simulation loop.
Zero overlap with graph neural network training for CAN bus intrusion detection.

---

## 9. ProxyStore (`proxystore/proxystore`)

**One-liner:** Transparent object proxy library for efficient large-object
transfer in distributed Python — avoids serialization overhead by passing
lazy references backed by pluggable stores (Redis, file, UCX, etc.).

| Aspect | Detail |
|--------|--------|
| Architecture | Proxy objects resolve on-demand from backend store (Redis, filesystem, Globus, etc.) |
| Install | `pip install proxystore` (MIT). Optional extras for Redis, endpoints |
| Python | 3.x |
| Maintenance | v1.0.0 (2026-04-02). 38 stars. Published at SC'23, IEEE TPDS 2024. UChicago/Argonne |

**Verdict: Not relevant now.** Solves inter-process large-object transfer.
GraphIDS runs single-node SLURM jobs with local data staging (NFS→scratch→TMPDIR).
No multi-node distributed training, no Dask clusters, no FaaS workflows.
Would become relevant with multi-node distributed training or Parsl-based sweeps
where checkpoint files pass between tasks.

---

## 10. Colmena (`exalearn/colmena`)

**One-liner:** Framework for AI-steered autonomous simulation campaigns on
supercomputers (Thinker/Doer pattern with pluggable workflow backends).

| Aspect | Detail |
|--------|--------|
| Architecture | Thinker (decision logic on head node) + Doer (task dispatch via Parsl/Globus Compute) |
| Install | `pip install colmena` (Apache-2.0) |
| Maintenance | v0.7.2 (2025-05-02). 60 stars. Argonne/DOE ExaLearn. Niche but active |

**Verdict: Not relevant.** Targets iterative simulation steering (run→analyze→decide
next run). GraphIDS has a fixed 3-stage pipeline, not an active-learning loop.

---

## 11. GlassBox (Globus Labs)

**One-liner:** Research umbrella for LLM interpretability — not an installable
library.

Collection of independent projects: AttentionLens (per-head decoder),
BalancedSubnet (memorization unlearning), Memory Injections (attention patching),
DART/TOXIN (toxicity intervention), LSHBloom (document dedup).

**Verdict: Not relevant.** Entirely focused on transformer/LLM interpretability.
No overlap with GNN training.

---

## 12. python-fire (`google/python-fire`)

**One-liner:** Auto-generates CLI from any Python object via introspection.
`fire.Fire(MyClass)` → every method becomes a subcommand.

| Aspect | Detail |
|--------|--------|
| Architecture | Single-call runtime introspection of function/class signatures |
| Install | `pip install fire` (Apache-2.0). Zero deps |
| Maintenance | 28k stars. Google-backed. v0.7.1 (2025-08-16). Actively maintained |

**vs. Typer (current):**

| Feature | Typer | python-fire |
|---------|-------|-------------|
| Type validation | Yes (type hints + Click) | No (loose coercion) |
| Help text | Rich, customizable | Auto-generated, minimal |
| Repeatable `--tla` flags | `list[str]` annotation | Awkward, no native support |
| Error messages | Typed, actionable | Raw Python tracebacks |
| Subcommand groups | `@app.command()` | Implicit from class hierarchy |

**Verdict: No benefit over Typer.** Fire's strength is zero-boilerplate
rapid prototyping. Typer's strength is production CLIs with validation,
help, and structure — exactly what GraphIDS needs. Switching would lose type
checking and the `--tla` repeatable-list pattern.

---

## 13. Globus Labs Projects (overview)

28 active + 8 archived projects. Only **Parsl** squarely addresses
SLURM-based ML training. Secondary relevance: **Globus Compute** (remote
submission), **ProxyStore** (distributed data flow), **Colmena** (steering
pattern). Everything else is domain-specific (materials, NLP, medical imaging,
compression) or infrastructure (data lifecycle, search, storage).

---
---

# Overall Summary

| Library | Category | OSC Compatible? | Relevant to GraphIDS? | Verdict |
|---------|----------|-----------------|----------------------|---------|
| **PySlurm** | SLURM bindings | Yes (Cython build) | Marginal (sacct only) | Not adopted — version coupling |
| **simple_slurm** | SLURM wrapper | Yes | No (no sacct) | Not adopted — wrong problem |
| **pyslurmutils** | SLURM REST | **No** (no slurmrestd) | N/A | Blocked |
| **slurm-pipeline** | SLURM pipeline | Untested on 3.12 | No | Not adopted — stale, heavy deps |
| **Parsl** | Workflow engine | Yes | **Future sweeps** | Best candidate if sweeps needed |
| **Globus Compute** | FaaS | Yes (needs internet) | No (single-cluster) | Not needed |
| **Garden AI** | Model publishing | Yes | No (training, not publishing) | Wrong phase |
| **Cascade** | Molecular dynamics | N/A | No (wrong domain) | Not relevant |
| **ProxyStore** | Data transfer | Yes | No (single-node jobs) | Not needed now |
| **Colmena** | Simulation steering | Yes | No (fixed pipeline) | Not relevant |
| **GlassBox** | LLM interpretability | N/A | No (wrong domain) | Not relevant |
| **python-fire** | CLI | Yes | No (Typer is better) | Not adopted |

**Libraries worth monitoring:**
- **Parsl** — if GraphIDS scales to large hyperparameter sweeps across many SLURM jobs
- **PySlurm** — if the accounting layer grows beyond 5 subprocess calls
- **submitit** (Meta) — if moving to programmatic job submission from Python

---
---

# Coverage Matrix: Handrolled vs. PySlurm vs. Parsl

Current `graphids/slurm/` = 904 Python + 145 shell = 1,049 lines.

## What we have

`✅` = covers this, `❌` = not covered, `🔶` = partial/different model

| Capability | Type | Handrolled | LOC | PySlurm | Parsl |
|-----------|------|-----------|-----|---------|-------|
| **A. Env var reads** | Config | ✅ `env.py` | 39 | ❌ | ❌ |
| **B. Resource spec model** | Data model | ✅ Pydantic `ResourceSpec` | 355 | ❌ | ❌ |
| **C. Profile lookup** (type×scale×stage) | Config resolution | ✅ JSON-driven `get_resources()` | (in B) | ❌ | ❌ |
| **D. Cluster auto-detection** | Config resolution | ✅ hostname → partition/gres | (in B) | 🔶 `Partition`/`Node` API | ❌ |
| **E. Dataset time scaling** | Config resolution | ✅ per-dataset multipliers | (in B) | ❌ | ❌ |
| **F. Failure-reaction scaling** | Retry policy | ✅ OOM→1.5× mem, TIMEOUT→1.5× time | (in B) | ❌ | 🔶 block retry (not resource scaling) |
| **G. Resource override validation** | Validation | ✅ reject unknown keys | (in B) | ❌ | ❌ |
| **H. Submit profile printer** | Shell integration | ✅ stdout line for `submit.sh` | (in B) | ❌ | ❌ |
| **I. sacct query** | Data read | ✅ `sacct_query()`, `sacct_by_user()` | 85 | ✅ `db.Job`/`db.JobFilter` | ❌ |
| **J. Job postmortem** (wall+RSS) | Data read + parse | ✅ parent+batch row merge | (in I) | ✅ `db.JobStatistics` | ❌ |
| **K. Resource profiler** (efficiency%) | Analysis + display | ✅ mem%/CPU% + right-sizing recs | 245 | 🔶 raw data only, no recs | ❌ |
| **L. Job name parser** | Data parse | ✅ regex → stage/dataset/seed | (in K) | ❌ | ❌ |
| **M. Job discovery** (since date) | Data read | ✅ `discover_jobs_since()` | (in K) | ✅ `db.JobFilter(start_time=)` | ❌ |
| **N. Data staging** | I/O orchestration | ✅ 3-tier NFS→scratch→TMPDIR | 180 | ❌ | 🔶 `worker_init` shell string |
| **O. Job submission** | Execution | ✅ `submit.sh` | 53 | ✅ `JobSubmitDescription.submit()` | ✅ `SlurmProvider` auto-submits |
| **P. Env bootstrap** | Shell setup | ✅ `_preamble.sh` | 47 | ❌ | 🔶 `worker_init` shell string |
| **Q. Post-job accounting** | Shell cleanup | ✅ `_epilog.sh` | 16 | ❌ | ❌ |

**Project-specific (A–H, L, N–Q):** ~670 lines. No library replaces these.
**Generic SLURM (I, J, K, M):** ~280 lines. PySlurm covers the data read/parse;
our analysis + display logic (right-sizing recs, job name parsing) is ours alone.

## Full Type Cross-Reference (API-verified)

Verified against PySlurm 24.11 source (`*.pyx`) and Parsl source + docs.

| Type | Capabilities | PySlurm | Parsl | Handrolled |
|------|-------------|---------|-------|-----------|
| **Config** | A. Env var reads | 🔶 `load_environment()` reads `PYSLURM_JOBDESC_*` — not project env vars | ❌ | ✅ |
| **Config (cluster)** | *Read slurmctld config* (100+ properties, cgroup, MPI, acct_gather) | ✅ `slurmctld.Config.load()` — read-only | ❌ | ❌ |
| **Data model** | B. Resource spec model | ❌ | ❌ | ✅ |
| **Config resolution** | C. Profile lookup, D. Cluster detection, E. Dataset scaling | 🔶 D only — `Node.configured_gres`, `Partition.nodes` give raw data, no profile abstraction | ❌ | ✅ |
| **Validation** | G. Resource override validation | 🔶 `_validate_options()`: mutual exclusivity (mem_per_cpu vs mem_per_node vs mem_per_gpu), script format, CPU freq, node counts. No partition/account/feasibility check | 🔶 `@typeguard.typechecked` on Config + unique executor labels. No cross-component validation | ✅ |
| **Shell integration** | H. Submit profile printer, P. Env bootstrap, Q. Post-job cleanup | ❌ | 🔶 P only (`worker_init` raw string) | ✅ |
| **Data read (completed)** | I. sacct query, J. Job postmortem, M. Job discovery | ✅ `db.Job`, `db.JobFilter`, `db.JobStatistics` | ❌ | ✅ |
| **Data read (live)** | *Node utilization, free GPUs* | ✅ `Node.free_memory`, `idle_cpus`, `allocated_gres` vs `configured_gres`, `cpu_load`, `current_watts` | ❌ | ❌ |
| **Data parse** | L. Job name parser | ❌ | ❌ | ✅ |
| **Analysis + display** | K. Resource profiler (efficiency% + right-sizing recs) | 🔶 raw data only, no analysis | 🔶 `MonitoringHub` → SQLite + Flask dashboard (`parsl-visualize`): per-task RSS/CPU/IO via psutil, Gantt charts, DAG viz. **No GPU. No right-sizing recs.** | ✅ |
| **Monitoring (live)** | *Live job stats, per-task resource tracking* | ✅ `Job.load_stats()` — real-time from slurmctld | ✅ per-task psutil (RSS, CPU time, disk I/O, ctx switches) to SQLite, 30s interval. **No GPU.** | 🔶 post-hoc sacct only |
| **I/O orchestration** | N. Data staging (3-tier) | ❌ | 🔶 8 staging providers (`NoOp`, `FTP`, `HTTP`, `Globus`, `Rsync`, `Zip`) + `File` URI abstraction + `DataManager`. But **single-hop only** — no multi-tier NFS→scratch→TMPDIR | ✅ |
| **Execution** | O. Submission, *auto-scaling*, *pilot-job* | 🔶 O only (`JobSubmitDescription.submit()`) | ✅ all three: `SlurmProvider` + `HighThroughputExecutor` + `Strategy` (simple/htex_auto_scale) with min/max blocks | ✅ submit.sh |
| **Orchestration** | *DAG-aware task scheduling* | ❌ | ✅ `AppFuture` data dependency resolution | ❌ (Monarch is sequential) |
| **Control** | *Cancel/hold/suspend/modify/signal* | ✅ `Job.cancel()/.hold()/.modify()/.send_signal()` | ❌ | ❌ (manual scancel) |
| **Retry policy** | F. Failure-reaction scaling, *fault-tolerant retry* | ❌ | 🔶 per-task retry (`Config(retries=N, retry_handler=fn)`), but no resource scaling (OOM→more mem) | ✅ |
| **Utilities** | *Time parsing, memory humanize, nodelist expansion* | ✅ `utils.ctime` (timestr↔secs), `utils.helpers` (humanize/dehumanize, expand_range_str, nodelist_from_range_str, uid↔name) | ❌ | 🔶 `parse_elapsed()` only |
| **Env propagation** | *Pass env vars / config to jobs* | ✅ `JobSubmitDescription(environment=dict)`, `load_environment()`, `load_sbatch_options()` | 🔶 `worker_init` string + 5 auto-set `PARSL_*` vars | ✅ `_preamble.sh` |

## What libraries do that we don't (updated)

| Our gap | Type | Who has it | Details | Worth adding? |
|---------|------|-----------|---------|--------------|
| **Cluster config read** | Config (cluster) | PySlurm `slurmctld.Config.load()` | 100+ read-only properties: memory limits, priority weights, scheduler params, cgroup config, MPI config | **No.** We use static JSON profiles. Don't need runtime slurmctld queries. |
| **Live node utilization** | Data read (live) | PySlurm `Node` | `free_memory`, `idle_cpus`, `allocated_gres` vs `configured_gres`, `cpu_load`, `current_watts` per node | **Maybe.** Useful for dynamic batch sizing or pre-sweep "is the cluster busy?" checks. |
| **Submission validation** | Validation | PySlurm `_validate_options()` | Mutual exclusivity (mem_per_cpu vs mem_per_node vs mem_per_gpu), script format, CPU freq, node count sanity | **No.** Our Pydantic `validate_config()` + jsonnet catches these at config layer before any SLURM interaction. |
| **Per-task monitoring + dashboard** | Analysis + display | Parsl `MonitoringHub` + `parsl-visualize` | SQLite DB, per-task RSS/CPU/IO via psutil at 30s intervals, Flask dashboard with Gantt charts, DAG viz, resource usage views. **No GPU metrics.** | **Maybe.** We have OTel + DeviceStatsMonitor for GPU. Parsl's CPU/memory dashboard is the gap — but only matters with many parallel tasks. |
| **Job control from Python** | Control | PySlurm `Job.cancel()/.hold()/.modify()/.send_signal()` | Extend walltime, hold/release, signal running jobs, modify resource limits mid-flight | **Low.** Manual scancel works. Auto-hold on failure could be nice but not critical. |
| **Time/memory utils** | Utilities | PySlurm `utils.ctime`, `utils.helpers` | `timestr_to_secs()`, `humanize()`/`dehumanize()`, `nodelist_from_range_str()`, `expand_range_str()`, `uid_to_name()` | **Low.** We have `parse_elapsed()` and `ResourceSpec.mem_mb`/`time_minutes`. Could vendor the 2-3 useful helpers without the full library. |
| **File staging (single-hop)** | I/O orchestration | Parsl `DataManager` + 8 providers | `File` URI abstraction, `Globus`/`Rsync`/`HTTP`/`FTP` staging. Single-hop per file. | **No.** Our 3-tier staging is project-specific and multi-hop. Parsl's model doesn't replace it. |
| **Auto-scaling worker pools** | Execution | Parsl `Strategy` + `SlurmProvider` | Reactive block scaling (min/max bounds, parallelism ratio, idle timeout) | **Future.** Relevant when sweep matrix grows large. |
| **Pilot-job model** | Execution | Parsl `HighThroughputExecutor` | One SLURM job serves many tasks via ZMQ workers | **Future.** Same use case as auto-scaling. |
| **DAG-aware scheduling** | Orchestration | Parsl `AppFuture` graph | Automatic dependency resolution from data flow | **No.** Monarch runs stages sequentially. DAG is trivial. |
| **Declarative task retry** | Retry policy | Parsl `Config(retries=N, retry_handler=fn)` | Per-task retry with custom handler. No resource scaling. | **Maybe.** Our `run_chain()` retry loop works but is manual. |
| **Env propagation to jobs** | Env propagation | PySlurm `environment=dict` | Structured dict of env vars, `load_environment()` from `PYSLURM_JOBDESC_*`, `load_sbatch_options()` from script | **No.** `_preamble.sh` + `.env` sourcing handles this. |
| **Multi-site execution** | Execution | Parsl (AWS, GCE, K8s, multi-SLURM) | Execute across clusters/clouds | **No.** Single-cluster. |

