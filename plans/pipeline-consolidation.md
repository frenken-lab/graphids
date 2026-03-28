# Pipeline Consolidation Plan

> Status: **proposed** | Date: 2026-03-27

Replace scattered orchestration code with Lightning-native job configuration and a thin
DAG orchestrator. Two layers, clearly separated: Lightning owns everything inside a job,
a ~80-line orchestrator owns everything across jobs.

## Problem statement

9/10 jobs fail. The knobs that control a job are scattered across:
- `cluster.py` (74 lines, builds sbatch scripts)
- `_preamble.sh` (82 lines, env setup + data staging + GPU sampler + USR1 trap)
- `_epilog.sh` (57 lines, GPU report + sacct)
- Deleted `resources.yaml` (referenced in docs, doesn't exist)
- Deleted `pipeline/` package (stages, runner, manifest — all gone)
- Config dataclasses in `schema.py`
- Hardcoded values in DataModules and model constructors

There is no single surface that controls a job. No retry logic. No automatic resume.
No resource visibility. The orchestration layer has been rewritten multiple times
(custom stages → Dagster → manifest.py → cluster.py) without solving the fundamental
problem: Lightning already handles everything inside a job, and the orchestrator should
be thin.

## Architecture

### Layer 1: Within a job — one YAML file is the ONLY knob surface

Everything that controls a single training run goes in ONE YAML config that LightningCLI
reads. No Python code configures the Trainer — it's all declarative.

```yaml
# configs/stages/autoencoder_vgae_medium_set01.yaml

seed_everything: 42

model:
  class_path: graphids.core.models.vgae.VGAEModule
  init_args:
    vgae: {latent_dim: 64, conv_type: gatv2, heads: 4}
    training: {lr: 0.001, weight_decay: 1e-5, compile_model: false}

data:
  class_path: graphids.core.preprocessing.datamodule.CANBusDataModule
  init_args:
    dataset: set_01
    batch_size: 8192
    num_workers: 2
    dynamic_batching: true
    conv_type: gatv2
    heads: 4

optimizer:
  lr: 0.001
  weight_decay: 1e-5

lr_scheduler:
  T_max: 300

trainer:
  max_epochs: 300
  accelerator: gpu
  devices: 1
  precision: 16-mixed
  gradient_clip_val: 1.0
  log_every_n_steps: 50
  default_root_dir: experimentruns/dev/rf15/set_01/vgae_medium_autoencoder
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        save_top_k: 1
        save_last: true
        mode: min
        filename: "best_model"
    - class_path: pytorch_lightning.callbacks.EarlyStopping
      init_args: {monitor: val_loss, patience: 30, mode: min}
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    - class_path: pytorch_lightning.callbacks.DeviceStatsMonitor
  plugins:
    - class_path: pytorch_lightning.plugins.environments.SLURMEnvironment
      init_args: {auto_requeue: true}

# Resource hints — read by orchestrator, ignored by LightningCLI
slurm:
  partition: gpu
  gres: "gpu:1"
  time: "04:00:00"
  mem: "32G"
  cpus_per_task: 4
  signal: "B:USR1@300"
```

**The command to run any stage is always:**
```bash
srun python -m graphids fit --config configs/stages/<stage>.yaml
```

No custom Python runner. No custom training loop. Lightning does it all.

### What this one YAML replaces

| Current location | What it controls | YAML section |
|---|---|---|
| `configure_optimizers()` in 3 modules | Optimizer + scheduler | `optimizer:` + `lr_scheduler:` |
| Deleted `resources.yaml` | SLURM resources | `slurm:` |
| `_preamble.sh` CUDA config | GPU memory allocation | `trainer.precision` + env var |
| `_preamble.sh` GPU sampler | GPU utilization monitoring | `DeviceStatsMonitor` callback |
| `_preamble.sh` USR1 trap | Preemption handling | `SLURMEnvironment(auto_requeue: true)` |
| `_preamble.sh` data staging | Data to local SSD | `DataModule.prepare_data()` |
| Scattered checkpoint logic | Save/resume policy | `ModelCheckpoint` callback |
| `cluster.py` `_build()` | sbatch directives | `slurm:` section |
| Hardcoded `EarlyStopping` | When to stop | `EarlyStopping` callback |

### What `_preamble.sh` becomes

From 82 lines to ~10:

```bash
#!/bin/bash
# _preamble.sh — minimal: only what Lightning can't do (shell env setup)
module load python/3.12
source /users/PAS2022/rf15/KD-GAT/.venv/bin/activate
source /users/PAS2022/rf15/KD-GAT/.env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

| Deleted from `_preamble.sh` | Why |
|---|---|
| `scripts/data/stage_data.sh` (data staging) | `DataModule.prepare_data()` — runs on rank 0 only, idempotent |
| `nvidia-smi -l 30` background sampler | `DeviceStatsMonitor` callback in YAML |
| USR1 trap + signal forwarding (~20 lines) | `SLURMEnvironment(auto_requeue=true)` handles natively |
| `KD_GAT_STAGE_DIR` TMPDIR setup | `DataModule.prepare_data()` manages staging |

### What `_epilog.sh` becomes

`DeviceStatsMonitor` replaces the GPU sampler + awk summary. `sacct` one-liner stays.
Log rotation stays. HF push (currently broken — script deleted) either gets fixed or removed.

~15 lines instead of 57.

### Failure mode handling

| Failure | Current | Lightning-native |
|---|---|---|
| CUDA OOM | `OOMSkipMixin` skips batch | Same + `gradient_clip_val` + `precision: 16-mixed` in YAML |
| Wall-time preemption | Shell USR1 trap (fragile) | `SLURMEnvironment(auto_requeue=true)` — saves ckpt, requeues |
| Checkpoint corruption | Nothing | `ModelCheckpoint(save_last=true)` keeps fallback |
| Worker crash | Nothing | `persistent_workers=True` + spawn (DataModule); Lightning restarts epoch |
| NFS race on data cache | Shell marker + `flock` | `prepare_data()` — Lightning calls rank 0 only, then barrier |
| Job failure (any) | Manual resubmit, starts over | Orchestrator resubmits same config — `Trainer.fit()` auto-resumes from last checkpoint |

Evidence: Lightning docs `clouds/cluster_advanced.rst` — `SLURMEnvironment` auto-requeue;
`deploy/production_basic.rst` — `BasePredictionWriter` + checkpoint resume.

## Layer 2: Across jobs — thin DAG orchestrator

The orchestrator does NOT understand training. It only:
1. Reads `pipeline.yaml` for the DAG topology
2. Reads `slurm:` section from each stage's YAML config
3. Submits `python -m graphids fit --config <stage>.yaml` via sbatch
4. Chains dependencies via `--dependency=afterok:<upstream_job_id>`
5. Monitors via `sacct`, resubmits failures (Lightning auto-resumes from checkpoint)

### `graphids/orchestrate.py` (~80 lines)

```python
"""DAG orchestrator: submits LightningCLI stages to SLURM with dependency chaining."""

import subprocess
import yaml
from pathlib import Path

from graphids.config.constants import STAGE_DEPENDENCIES

SLURM_DEFAULTS = {
    "account": "PAS1266",
    "signal": "B:USR1@300",
}

def submit(config_path: Path, depends_on: list[int] | None = None) -> int:
    """Submit one LightningCLI stage to SLURM. Returns job ID."""
    cfg = yaml.safe_load(config_path.read_text())
    slurm = {**SLURM_DEFAULTS, **cfg.get("slurm", {})}

    sbatch_args = [
        "sbatch",
        f"--partition={slurm['partition']}",
        f"--time={slurm['time']}",
        f"--mem={slurm['mem']}",
        f"--cpus-per-task={slurm.get('cpus_per_task', 4)}",
        f"--signal={slurm['signal']}",
        f"--account={slurm['account']}",
        f"--output=slurm_logs/{config_path.stem}_%j.out",
        f"--error=slurm_logs/{config_path.stem}_%j.err",
    ]
    if slurm.get("gres"):
        sbatch_args.append(f"--gres={slurm['gres']}")
    if depends_on:
        dep_str = ":".join(str(d) for d in depends_on)
        sbatch_args.append(f"--dependency=afterok:{dep_str}")

    script = (
        "#!/bin/bash\\n"
        "source scripts/slurm/_preamble.sh\\n"
        f"srun python -m graphids fit --config {config_path}\\n"
        "source scripts/slurm/_epilog.sh\\n"
    )
    result = subprocess.run(
        [*sbatch_args, "--wrap", script],
        capture_output=True, text=True, check=True,
        cwd="/users/PAS2022/rf15/KD-GAT",
    )
    job_id = int(result.stdout.strip().split()[-1])
    return job_id


def run_pipeline(config_dir: Path, stages: list[str] | None = None):
    """Walk the DAG in topological order, submit stages with dependency chaining."""
    from graphids.config.constants import DEFAULT_STAGES, topo_sort

    stages = stages or DEFAULT_STAGES
    ordered = topo_sort(stages)
    job_ids: dict[str, int] = {}
    for stage in ordered:
        config_path = config_dir / f"{stage}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config for stage '{stage}' at {config_path}")
        deps = [job_ids[d] for d in STAGE_DEPENDENCIES.get(stage, []) if d in job_ids]
        job_ids[stage] = submit(config_path, depends_on=deps or None)
    return job_ids


def check_status(job_id: int) -> str:
    """Query SLURM job status via sacct."""
    result = subprocess.run(
        ["sacct", "-j", str(job_id), "--format=State", "--noheader", "--parsable2"],
        capture_output=True, text=True,
    )
    states = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    return states[0] if states else "UNKNOWN"


def retry_failed(job_ids: dict[str, int], config_dir: Path, max_retries: int = 2):
    """Check job statuses, resubmit failed stages.

    Lightning auto-resumes from last checkpoint — resubmission is free.
    """
    for stage, jid in list(job_ids.items()):
        status = check_status(jid)
        if status in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"):
            config_path = config_dir / f"{stage}.yaml"
            new_id = submit(config_path)
            job_ids[stage] = new_id
    return job_ids
```

### Usage

```bash
# Submit full pipeline:
python -m graphids.orchestrate configs/ablation_run_004/

# Submit specific stages:
python -m graphids.orchestrate configs/ablation_run_004/ --stages autoencoder curriculum

# Check and retry failures:
python -m graphids.orchestrate configs/ablation_run_004/ --retry
```

### Retry semantics

Resubmission is free because of Lightning's checkpoint contract:
1. `ModelCheckpoint(save_last=True)` writes `last.ckpt` every epoch
2. `SLURMEnvironment(auto_requeue=True)` saves on preemption signal
3. `Trainer.fit()` auto-detects `last.ckpt` in `default_root_dir` and resumes
4. No custom resume logic needed — the YAML config is identical on retry

For OOM failures specifically, the orchestrator can optionally adjust the config before
resubmit (reduce batch_size, increase mem) — but the first fix should be getting the
resource profiles right in the YAML to begin with.

## Config generation

For ablation sweeps, configs are generated rather than hand-written. A small generator
reads `pipeline.yaml` (DAG topology) + `presets.yaml` (model configs) and emits one
YAML per stage×dataset×config:

```bash
# Generate all configs for an ablation:
python -m graphids.generate_configs \
  --ablation configs/ablation_run_004.yaml \
  --output-dir configs/ablation_run_004/
```

This replaces the deleted `ablation.yaml` + `manifest.py`. The generator is ~50 lines:
- Read ablation spec (which configs × which datasets × which seeds)
- For each combo, emit a stage YAML with the right model/data/trainer/slurm sections
- Dedup shared upstream stages (e.g., one VGAE autoencoder shared by multiple GAT runs)

## Orchestration decision: restore Dagster architecture, no daemon

### History

Dagster was integrated at `b2e8845` (2026-03-19) with a solid architecture:
`dagster_defs.py` (271 lines) + `pipes_slurm.py` (123 lines) + `slurm_primitives.py`
(245 lines). It was deleted one day later at `ff176c4` as collateral in the LightningCLI
migration sprint — not because it was broken, but because it depended on the old config
system (`resolve()`, `PipelineConfig.variants`) which was replaced.

### What the Dagster code had (worth restoring)

| Component | Source | What it does |
|---|---|---|
| `RESOURCE_PROFILES` | `resources.yaml` → `slurm_primitives.py` | `(model, scale, stage) → ResourceSpec` — centralized resource config |
| `FAILURE_REACTIONS` + `scale_resources()` | `resources.yaml` + `slurm_primitives.py` | Adaptive retry: OOM → 2× mem, TIMEOUT → 1.5× time |
| `build_dag_topology()` | `dagster_defs.py:171-208` | Dynamic DAG from `STAGE_DEPENDENCIES` + config variants |
| `fire_and_forget()` | `dagster_defs.py:215-241` | Zero-daemon sbatch chaining with `--dependency=afterok` |
| Multi-partition `(dataset, seed)` | `dagster_defs.py:34-37` | Matrix submission across datasets × seeds |
| `PipesSlurmClient` | `pipes_slurm.py` | Dagster Pipes over NFS (no SSH/S3) |

### Approach: Option A — Dagster as library, no daemon

Use Dagster's architecture (DAG construction, resource profiles, retry policies,
partitions) but execute via `fire_and_forget` + a polling loop on a CPU job.
No webserver, no daemon on HPC.

**Why not the daemon (Option B):**
- `45c9bc6` tried "Dagster daemon as SLURM job" — adds a layer of fragility
  (daemon job must stay alive for full pipeline duration)
- Login nodes can't run persistent services
- The `fire_and_forget` mode already existed and chains sbatch dependencies correctly

**Dagster UI optionally available** via `dagster dev` on local machine (WSL) pointing
at the NFS-mounted run directory for visualization. Not required for execution.

### Restored orchestrator architecture (~300 lines total)

```
graphids/orchestrate/
  __init__.py           # CLI entry point
  dag.py                # build_dag_topology(), topo_sort (~80 lines)
                        # Restored from b2e8845:dagster_defs.py:171-208
  resources.py          # RESOURCE_PROFILES, FAILURE_REACTIONS, scale_resources (~60 lines)
                        # Restored from b2e8845:slurm_primitives.py
  submit.py             # fire_and_forget(), submit_one(), poll_and_retry (~100 lines)
                        # Restored from b2e8845:dagster_defs.py:215-241 + new poll loop
  generate_configs.py   # Ablation spec → per-stage YAML configs (~50 lines)
```

**`resources.yaml` (restored):**
```yaml
resource_profiles:
  vgae:
    medium:
      autoencoder: {partition: gpu, gres: "gpu:1", time: "04:00:00", mem: "32G", cpus: 4}
      curriculum:  {partition: gpu, gres: "gpu:1", time: "04:00:00", mem: "32G", cpus: 4}
    large:
      autoencoder: {partition: gpu, gres: "gpu:1", time: "06:00:00", mem: "48G", cpus: 4}
  gat:
    medium:
      normal:      {partition: gpu, gres: "gpu:1", time: "04:00:00", mem: "32G", cpus: 4}
      curriculum:  {partition: gpu, gres: "gpu:1", time: "04:00:00", mem: "32G", cpus: 4}
  dqn:
    medium:
      fusion:      {partition: gpu, gres: "gpu:1", time: "02:00:00", mem: "16G", cpus: 4}

failure_reactions:
  OUT_OF_MEMORY:
    scale_mem: 2.0
    max_retries: 2
  TIMEOUT:
    scale_time: 1.5
    max_retries: 1
  NODE_FAIL:
    max_retries: 2
```

**`submit.py` core loop — interactive orchestrator:**
```python
def run_pipeline(config_dir: Path, datasets: list[str], seeds: list[int],
                 max_retries: int = 2, poll_interval: int = 300):
    """Long-running orchestrator: submit stages as deps complete, poll, retry.

    Submits GPU jobs one at a time as upstream dependencies complete. Safe to
    Ctrl+C — running GPU jobs continue independently. Restart the orchestrator
    and it picks up by checking sacct for already-completed stages.
    """
    dag = build_dag_topology()
    order = topo_sort(dag)

    for dataset in datasets:
        for seed in seeds:
            log.info("pipeline_start", dataset=dataset, seed=seed)
            job_ids: dict[str, int] = {}
            retries: dict[str, int] = {}

            # Check for already-completed stages (orchestrator restart)
            completed = _scan_completed(config_dir, dag, dataset, seed)

            for name, node in order:
                if name in completed:
                    log.info("already_complete", stage=name)
                    continue

                # Wait for upstream deps to complete
                for dep in node.deps:
                    if dep in job_ids:
                        _wait_for_job(job_ids[dep], dep, poll_interval)

                # Submit this stage
                resources = get_resources(node.resource_model, node.scale, node.stage)
                config_path = config_dir / f"{name}.yaml"
                job_id = submit_one(config_path, resources)
                job_ids[name] = job_id
                log.info("submitted", stage=name, job_id=job_id,
                         resources=resources, deps=list(node.deps))

                # Poll until this stage completes (or fails)
                status = _wait_for_job(job_id, name, poll_interval)

                if status in ("FAILED", "OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL"):
                    if retries.get(name, 0) < max_retries:
                        resources = scale_resources(resources, status)
                        new_id = submit_one(config_path, resources)
                        job_ids[name] = new_id
                        retries[name] = retries.get(name, 0) + 1
                        log.info("retried", stage=name, reason=status,
                                 new_job=new_id, retry=retries[name])
                        # Re-wait for the retry
                        _wait_for_job(new_id, name, poll_interval)
                    else:
                        log.error("max_retries_exhausted", stage=name, status=status)
                        # Don't submit downstream stages

            log.info("pipeline_done", dataset=dataset, seed=seed,
                     completed=len(job_ids), retried=sum(retries.values()))


def _wait_for_job(job_id: int, name: str, poll_interval: int) -> str:
    """Block until job reaches terminal state. Returns status string."""
    while True:
        status = check_status(job_id)
        if status in ("COMPLETED", "FAILED", "OUT_OF_MEMORY",
                       "TIMEOUT", "NODE_FAIL", "CANCELLED"):
            log.info("job_terminal", stage=name, job_id=job_id, status=status)
            return status
        time.sleep(poll_interval)


def _scan_completed(config_dir, dag, dataset, seed) -> set[str]:
    """Check which stages already completed (checkpoint + metrics exist).

    Enables orchestrator restart without resubmitting finished work.
    """
    completed = set()
    for name, node in dag.items():
        run_dir = _resolve_run_dir(config_dir, name, dataset, seed)
        if (run_dir / "best_model.ckpt").exists():
            completed.add(name)
    return completed
```

**Usage — long-running interactive CPU job:**
```bash
# Start an interactive orchestrator session (can intervene, pause, adjust):
srun --partition=cpu --time=24:00:00 --mem=4G --cpus-per-task=1 --account=PAS1266 \
  --pty bash -c "source scripts/slurm/_preamble.sh && python -m graphids.orchestrate \
    --config-dir configs/ablation_run_004/ \
    --datasets set_01 set_02 \
    --seeds 42"

# Or via sbatch with output you can tail:
sbatch --partition=cpu --time=24:00:00 --mem=4G --cpus-per-task=1 --account=PAS1266 \
  --output=slurm_logs/orchestrate_%j.out \
  --wrap="source scripts/slurm/_preamble.sh && python -m graphids.orchestrate \
    --config-dir configs/ablation_run_004/ --datasets set_01 set_02 --seeds 42"
# Then: tail -f slurm_logs/orchestrate_<job_id>.out
```

**No `fire_and_forget`.** The orchestrator is always a long-running poll loop on a CPU
job. This way you can:
- Watch progress in real time (`tail -f` or `--pty`)
- Intervene: cancel specific stages, adjust resources, skip stages
- Not blow away an entire pipeline when one thing needs changing
- Ctrl+C the orchestrator without affecting running GPU jobs (they have their own sbatch)

The orchestrator submits GPU jobs one at a time as dependencies complete, rather than
chaining everything upfront. If you kill the orchestrator, running GPU jobs continue —
you just restart the orchestrator and it picks up where it left off by checking `sacct`.

**Optional: Dagster UI on WSL for visualization (not required):**
```bash
# dagster dev -m graphids.orchestrate.dagster_viz
```

### How it connects to Lightning

Each submitted job runs:
```bash
srun python -m graphids fit --config configs/stages/<stage>.yaml
```

Lightning handles everything inside the job. The orchestrator only cares about:
- Did the job complete? (`sacct` exit code)
- If not, what failed? (OOM / TIMEOUT / NODE_FAIL)
- Resubmit with scaled resources — Lightning auto-resumes from `last.ckpt`

## Spike test: SLURMEnvironment verification

Submitted as job `46012629` on `gpu` partition (`tests/spikes/spike_slurm_lightning.py`).

Tests:
- `SLURMEnvironment(auto_requeue=True)` detects SLURM
- `ModelCheckpoint(save_last=True)` writes `last.ckpt`
- USR1 signal at 90s before wall-time triggers checkpoint + requeue
- On requeue, `Trainer.fit(ckpt_path="last")` auto-resumes
- `DeviceStatsMonitor` logs GPU stats

**Results (job 46012629, 2026-03-27):**
- `SLURMEnvironment` detected SLURM, ran on `p0228` V100
- `ModelCheckpoint` wrote both `best_model.ckpt` and `last.ckpt`
- `DeviceStatsMonitor` ran without error
- 50 epochs in 7s, `val_loss=0.015`, exit code 0
- Job completed too fast to test USR1 auto-requeue (needs longer run for that)
- **Core contract verified: Lightning + SLURM + checkpoints + device monitoring works on Pitzer**

## What gets deleted

| File/Package | Lines | Why |
|---|---|---|
| `cluster.py` | 74 | Replaced by `orchestrate.py` |
| `_preamble.sh` (most of it) | ~70 | Lightning handles: data staging, GPU monitoring, USR1 trap |
| `_epilog.sh` (most of it) | ~40 | `DeviceStatsMonitor` replaces GPU sampler + awk |
| `graphids/__init__.py` "pipeline" lazy import | 2 | Package doesn't exist |
| Stale `pipeline.yaml` stage definitions | — | Stages are just LightningCLI configs now, not code |

## What gets created

| File | Lines | Source |
|---|---|---|
| `graphids/orchestrate/__init__.py` | ~20 | CLI entry point (interactive poll loop) |
| `graphids/orchestrate/dag.py` | ~80 | `build_dag_topology()`, topo sort (from `b2e8845`) |
| `graphids/orchestrate/resources.py` | ~60 | `RESOURCE_PROFILES`, `FAILURE_REACTIONS`, `scale_resources` (from `b2e8845`) |
| `graphids/orchestrate/submit.py` | ~100 | `fire_and_forget()`, `submit_one()`, `poll_and_retry()` (from `b2e8845` + new poll) |
| `graphids/orchestrate/generate_configs.py` | ~50 | Ablation spec → per-stage YAML configs |
| `graphids/config/defaults/resources.yaml` | ~50 | Resource profiles + failure reactions (restored) |
| `configs/stages/*.yaml` | ~50 each | One per stage×dataset×config (generated, data not code) |
| `_preamble.sh` (trimmed) | ~10 | Module load + venv + env vars only |
| `_epilog.sh` (trimmed) | ~15 | sacct + log rotation only |

## Execution order

1. Verify `SLURMEnvironment(auto_requeue=true)` on Pitzer (spike test `46012629` pending)
2. Write `resources.yaml` with resource profiles + failure reactions
3. Restore `dag.py` from `b2e8845` — adapt to current config system (no more `resolve()`)
4. Restore `resources.py` from `b2e8845:slurm_primitives.py` — `ResourceSpec` + scaling
5. Write `submit.py` — `run_pipeline()` interactive loop + `submit_one()` + `_wait_for_job()`
6. Write `generate_configs.py` — ablation spec → stage YAMLs
7. Write example stage YAML configs (one per stage type)
8. Wire `SLURMEnvironment(auto_requeue=true)` and `DeviceStatsMonitor` into config templates
9. Trim `_preamble.sh` to ~10 lines, `_epilog.sh` to ~15 lines
10. Delete `cluster.py`
11. Clean dead `"pipeline"` import from `graphids/__init__.py`
12. End-to-end test: submit a 2-stage pipeline (preprocess → autoencoder) on `hcrl_ch`
13. Full ablation run via orchestrator

## Line count

| Change | Lines |
|--------|-------|
| Delete `cluster.py` | -74 |
| Trim `_preamble.sh` | -70 |
| Trim `_epilog.sh` | -40 |
| Delete dead pipeline import | -2 |
| `graphids/orchestrate/` package | +310 |
| `resources.yaml` | +50 |
| **Net** | **+174 lines of code** |

This is a net add because we're restoring real functionality (adaptive retry,
resource profiles, DAG construction, multi-partition submission) that was deleted.
The custom code being added is domain-specific orchestration that no framework
provides — Lightning handles the job interior, this handles the job exterior.

(Plus ~50 lines per generated YAML config, but those are data, not code.)

## Risks

- **`SLURMEnvironment` auto-requeue on OSC Pitzer**: needs verification that SLURM's
  `scontrol requeue` is enabled for the PAS1266 account. Some clusters restrict requeue
  to specific partitions. Test with a short `gpudebug` job first.
- **`DeviceStatsMonitor` overhead**: polls `nvidia-smi` each step. If this adds measurable
  overhead, switch to logging every N steps via callback config, or fall back to the
  background sampler approach for the `_epilog.sh` summary only.
- **`Trainer.fit()` auto-resume**: relies on `last.ckpt` being in `default_root_dir`.
  If the run directory structure changes between retries (e.g., identity hash differs),
  the checkpoint won't be found. Pin `default_root_dir` in the YAML explicitly.
- **Config generation for KD stages**: KD configs need `teacher_ckpt_path` pointing to
  a completed upstream run. The generator must resolve this path from the DAG, not
  hardcode it.
- **`prepare_data()` for data staging**: the preprocessing plan (section 6a) must be
  completed first — currently data staging is in `_preamble.sh`, not in the DataModule.
  Until that's done, keep the `stage_data.sh` call in `_preamble.sh`.

## Dependencies on other plans

| This plan needs | From plan |
|---|---|
| `DataModule.prepare_data()` for data staging | Preprocessing plan section 6a |
| `configure_optimizers` deleted from modules | Models plan sections 1, 5 |
| `add_optimizer_args` in `GraphIDSCLI` | Models plan section 1 |
| Stage YAML configs referencing correct model class paths | Models plan (new class names) |
