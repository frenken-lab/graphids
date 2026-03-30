# wandb Research for KD-GAT

> Last modified: 2026-03-30
> Context: PyTorch Lightning + PyG on OSC HPC (SLURM, V100). Config: jsonargparse (NOT Hydra/OmegaConf). 18 ablation configs × 2 datasets × seeds. Currently `logger: false`.

## Decision Summary

| Area | Verdict | Reason |
|------|---------|--------|
| Tracking & Logging | **ADOPT** | Auto GPU metrics, Lightning integration, works on OSC online |
| Sweeps | **SKIP** | Conflicts with dagster orchestration; Optuna superior for HPC |
| Model Registry | **SKIP** | Single researcher; filesystem + DuckDB sufficient; saves storage quota |
| Data Artifacts | **SKIP** | No local FS reference support; existing staging protocol works |
| dagster-wandb | **SKIP** | Training in SLURM subprocess, not dagster process |
| Pricing | **Non-issue** | Academic 100 GB free; ~200-400 MB for 36 runs |

---

## Adoption History (git archaeology)

**This is the third attempt at experiment tracking.**

| Period | Tracker | What happened |
|--------|---------|---------------|
| Feb 17 – Mar 6, 2026 | wandb | Custom Python wrappers (`_init_wandb`, `_finish_wandb`, `_wandb_log_metrics` in cli.py), WandbLogger in trainer_factory.py, offline mode SLURM detection, post-job sync in _epilog.sh. ≥10 production offline runs recorded. |
| ~Mar 7 | MLflow | Replaced wandb (`5cb4faf`). Intended as simpler alternative. |
| Mar 19 | Nothing | MLflow removed (`be996b8`) for dependency bloat (~50MB + 30 transitive deps). |
| Mar 19 – present | Nothing | `logger: false` in trainer.yaml. Zero experiment tracking. |

**Why wandb was removed:** Not technical failure — it worked. Removed during aggressive March 2026 simplification campaign. wandb coexisted with MLflow (4 sync mechanisms → 2). Both cut to reduce complexity.

**What's different this time:**
1. YAML-only config via LightningCLI — no custom Python wrappers (the wrappers were the complexity that got cut)
2. No competing tracker (MLflow gone)
3. jsonargparse handles logger lifecycle natively
4. Clear separation: wandb in Lightning layer only, dagster manages orchestration only

**Lesson to document:** The previous `_init_wandb()` / `_finish_wandb()` / `_wandb_log_metrics()` custom code was the problem, not wandb itself. LightningCLI eliminates all of it — WandbLogger is just a YAML config entry.

---

## jsonargparse vs OmegaConf Conflict

### The problem is silent corruption, not an error

jsonargparse 4.47.0 in `parser_mode="yaml"` treats `${oc.env:KD_GAT_LAKE_ROOT,experimentruns}` as a **literal string**. It passes `"${oc.env:KD_GAT_LAKE_ROOT,experimentruns}"` to `WandbLogger(save_dir=...)`, creating a directory with that garbage name. No error, no warning. (Empirically verified in project venv.)

### jsonargparse env var support (already wired)

`graphids/cli.py` already configures `default_env=True` and `env_prefix="KD_GAT"`. Any config key can be overridden via env var:
```
KD_GAT_FIT__TRAINER__LOGGER__INIT_ARGS__SAVE_DIR=/path/to/dir
```
This works but is verbose. Better solution below.

### OmegaConf cannot coexist

Even if installed, `parser_mode="omegaconf"` only resolves interpolation within a single YAML file. KD-GAT's multi-file chain (trainer.yaml → stage YAML → overlay YAML) would break cross-file refs. Project rules explicitly forbid OmegaConf. **Hard either/or, no adapter.**

### Correct WandbLogger YAML for jsonargparse

```yaml
# trainer.yaml — works with jsonargparse, no OmegaConf
trainer:
  logger:
    - class_path: pytorch_lightning.loggers.WandbLogger
      init_args:
        project: kd-gat
        save_dir: null  # resolved via link_arguments (see below)
        log_model: false
        tags: null  # set via CLI override per run
      dict_kwargs:
        group: null  # set via CLI override per run
    - class_path: pytorch_lightning.loggers.CSVLogger
      init_args:
        save_dir: null  # resolved via link_arguments
```

### Recommended solution: static path or WANDB_DIR env var

`link_arguments` is unnecessary here — it can target `trainer.logger.init_args.save_dir` (no namespace restriction), but wandb logs and training checkpoints don't need the same directory.

**Option A — Static path in YAML (simplest):**
```yaml
save_dir: /fs/ess/PAS1266/kd-gat/wandb  # ESS path doesn't change
```

**Option B — WANDB_DIR in _preamble.sh (preferred):**
```bash
export WANDB_DIR=/fs/scratch/PAS1266/wandb  # wandb reads this natively
```
Bypasses `save_dir` entirely. Set once in shell, wandb respects it. Scratch is better I/O than ESS for write-heavy wandb logs.

---

## Tracking & Logging (ADOPT)

### Automatic system metrics (zero code, 15s interval)

- GPU: utilization %, memory allocated/used, temperature (°C), power (W), clock speeds
- CPU: process CPU %, thread count, RSS, system memory %
- Disk: usage %, read/write throughput
- Network: bytes sent/received

Collection via vendored pynvml (lightweight C calls, not nvidia-smi subprocess). Polling interval configurable via `WANDB__STATS_SAMPLING_INTERVAL` env var. Overhead: negligible (<1% training time per community reports).

### save_hyperparameters() + LightningCLI gotcha

**Known issue ([Lightning #19728](https://github.com/Lightning-AI/pytorch-lightning/issues/19728)):** Full LightningCLI config is NOT automatically forwarded to wandb config. Fix: ~10-line callback in cli.py:

```python
class WandbSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer, pl_module, stage):
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.config.update(self.config.as_dict())
        super().save_config(trainer, pl_module, stage)
```

### HPC/SLURM handling

Online mode works on OSC (outbound HTTPS confirmed). Key env vars for `_preamble.sh`:
- `WANDB_DIR=/fs/scratch/PAS1266/wandb` — run data on scratch (better I/O than NFS)
- `WANDB_DISABLE_GIT=true` — skip git probing (NFS perf)
- `WANDB_SILENT=true` — reduce stdout noise in SLURM logs
- `WANDB_MODE=offline` — fallback only if network flakes; sync via `wandb sync` in epilog

### Multi-run comparison

Parallel coordinates, run grouping by config fields, MongoDB-style filters, side-by-side config diff. 18 ablation configs × 2 datasets can group by claim (loss × curriculum, fusion method, conv type).

---

## Sweeps: wandb vs Optuna (SKIP wandb Sweeps)

| Feature | wandb Sweeps | Optuna |
|---------|-------------|--------|
| Algorithms | Random, Grid, Bayes (GP) | TPE, CMA-ES, GP, NSGA-II multi-objective |
| Offline | **NOT supported** | Fully offline (SQLite) |
| SLURM | Agent per job, needs internet | DB on shared FS, no internet needed |
| Orchestrator conflict | Yes (agent is launcher → fights dagster) | No (library, not launcher) |

wandb sweeps require the agent to be the process launcher. dagster already fills that role. They cannot coexist without one becoming a pass-through. For future HPO: Optuna + `WeightsAndBiasesCallback` logs trials to wandb for visualization.

---

## Model Registry, Data Artifacts, dagster-wandb (all SKIP)

**Model Registry:** Uploading checkpoints consumes 7-18 GB per ablation round of 100 GB quota. Existing `lake_root/{dataset}/{model_type}_{scale}_{stage}_{identity_hash}/seed_N/` + DuckDB catalog tracks lineage via `identity_hash`. Keep `log_model: false`.

**Data Artifacts:** `add_reference()` does not support local filesystem paths (`file://`), only S3/GCS/HTTPS. Data lives on ESS/scratch. Existing staging protocol + `preprocessing_version` constant handles versioning. Overkill.

**dagster-wandb ([docs](https://docs.dagster.io/integrations/libraries/wandb/dagster-wandb)):** Provides `wandb_artifacts_io_manager` that serializes Python objects as wandb Artifacts. Assumes training runs **in the dagster process**. KD-GAT submits SLURM jobs via sbatch — training is in a separate process. The IO Manager would create empty wandb runs in dagster. wandb belongs in the Lightning layer only. If dagster needs wandb results post-completion, query via `wandb.Api()` — future optimization.

---

## Pricing

Academic tier: 100 GB storage, all Pro features, free with .edu email. 36 runs at ~5-10 MB each = ~200-400 MB. Rate limit: 200 req/min (supports ~15 concurrent runs with auto-backoff). Storage exceeded = warning banner only, no lockout. All data exportable via API (DataFrames, CSV, Parquet).

---

## Implementation Checklist

1. `uv add wandb` + `wandb login` on login node (one-time, API key in `~/.netrc`)
2. WandbLogger + CSVLogger in `trainer.yaml` (YAML syntax above — static `save_dir` or rely on `WANDB_DIR`)
3. `WandbSaveConfigCallback` in `cli.py` (~10 lines, forwards jsonargparse config to wandb)
4. Env vars in `_preamble.sh` (`WANDB_DIR`, `WANDB_DISABLE_GIT`, `WANDB_SILENT`)
5. Optional: `wandb sync` in `_epilog.sh` for offline fallback

## Sources

- [System metrics reference](https://docs.wandb.ai/models/ref/python/experiments/system-metrics)
- [wandb offline docs](https://docs.wandb.ai/support/run_wandb_offline/)
- [wandb env vars](https://docs.wandb.ai/guides/track/environment-variables)
- [wandb pricing](https://wandb.ai/site/pricing/)
- [wandb sweeps on SLURM](https://docs.wandb.ai/support/run_sweeps_slurm/)
- [Optuna + wandb integration](https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.WeightsAndBiasesCallback.html)
- [dagster-wandb docs](https://docs.dagster.io/integrations/libraries/wandb/dagster-wandb)
- [LightningCLI + WandbLogger issue #19728](https://github.com/Lightning-AI/pytorch-lightning/issues/19728)
- [W&B Registry](https://docs.wandb.ai/models/registry)
- [W&B Artifacts](https://docs.wandb.ai/guides/artifacts/construct-an-artifact/)
