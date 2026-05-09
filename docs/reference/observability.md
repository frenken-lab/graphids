# Observability & Logging

> Last verified: 2026-05-08. Single sink: **MLflow SQLite** for all metrics;
> **`.parsl_scripts/`** for SLURM log text; **`checkpoints/`** for weights.

## Filesystem layout

```
$GRAPHIDS_LAKE_ROOT/mlflow.db                        ← all metrics / params / tags
$GRAPHIDS_RUN_ROOT/{dataset}/{subdir}/{group}/{variant}/seed_{N}/
  checkpoints/best_model.ckpt[.sha256]               ← top-val checkpoint
  checkpoints/last.ckpt[.sha256]                     ← resume checkpoint
  artifacts/                                         ← analyze-row outputs
  .parsl_scripts/{jobname}.{hash}.stderr             ← Lightning + SLURM logs
  .parsl_scripts/cmd_{jobname}.{hash}.sh             ← submitted sbatch script
```

## MLflow schema

**Two rows per run** sharing `run_name = {group}_{variant}_{dataset}_seed{N}`,
in experiment `graphids/{dataset}/{group}`, distinguished by `graphids.phase`.

### Identity tags (both rows)
```
graphids.phase / plan_id / plan_module / git_sha / row_name
graphids.run_dir / dataset / group / variant / seed / model_type / scale
slurm.job_id / slurm.cluster_name
```
Added at fit-end: `graphids.budget_binding`, `graphids.ckpt_path`, `graphids.ckpt_sha256`

### Params (fit row, logged once at epoch 0)
Model hyperparams (`conv_type`, `hidden_dims`, `latent_dim`, `heads`, `lr`, `weight_decay`, …)
+ dataloader params (`graphids.budget_target_bytes`, `graphids.num_workers`, …)

### Per-epoch metrics (fit row, `step = epoch`)
| key | who logs it |
|---|---|
| `train_loss`, `val_loss`, `epoch` | all |
| `train_acc`, `val_acc`, `val_auroc`, `lr-Adam` | GAT / fusion |
| `train_recon`, `train_canid`, `train_kl`, `val_discrimination_ratio` | VGAE |
| `graphids.peak_vram_mb` | single point at fit-end |

### System telemetry (both rows, 5 s interval, MLflow background thread)
```
system/cpu_utilization_percentage
system/system_memory_usage_{megabytes,percentage}
system/disk_{usage,available}_megabytes
system/gpu_0_{memory_usage_megabytes,memory_usage_percentage,utilization_percentage,power_usage_watts}
```

### Test metrics (test row, single point)
```
accuracy / ap / auroc / ece / f1 / mcc / precision / recall / specificity / threshold
test/precision_at_0.95recall  /  test/recall_at_0.99precision
test/{subtest}/auroc  /  test/{subtest}/auroc_per_attack/{attack_type}
test/{subtest}/auroc_per_attack_macro
```
Subtests: `test_01_known_vehicle_known_attack` … `test_06_masquerade`

## Querying

```python
from graphids._mlflow import configure_tracking_uri, build_search_filter
from mlflow.tracking import MlflowClient

configure_tracking_uri()
client = MlflowClient()
exp_ids = [e.experiment_id for e in client.search_experiments(
    filter_string="name LIKE 'graphids/%'"
)]

# Finished test runs for a dataset
runs = client.search_runs(exp_ids,
    filter_string=build_search_filter(dataset="set_01", phase="test", status="FINISHED"))

# Per-epoch history
hist = client.get_metric_history(run_id, "val_auroc")  # list of (step, value)

# All runs for a plan
runs = client.search_runs(exp_ids,
    filter_string=build_search_filter(plan_id="019e05a9-..."))
```

**MLflow UI:**
```bash
source .env
mlflow ui --backend-store-uri "sqlite:///$GRAPHIDS_LAKE_ROOT/mlflow.db" --port 5000
# SSH tunnel: ssh -L 5000:localhost:5000 pitzer  →  http://localhost:5000
```

**SLURM logs:**
```bash
ls -t {run_dir}/.parsl_scripts/*.stderr | head -1 | xargs tail -50
gx plans where <plan_id> --row <row_name>   # prints run_dir + stderr path
```

## LoggedModel (checkpoint index)

Each fit creates a metadata-only MLflow `LoggedModel` (no artifact bytes):
tags carry `graphids.ckpt_path` + `graphids.ckpt_sha256`; downstream rows
resolve via `client.search_logged_models(...)` then read `lm.tags["graphids.ckpt_path"]`.

## GPU profiling tools

| Tool | Use for |
|---|---|
| `graphids.peak_vram_mb` + `system/gpu_0_*` in MLflow | automatic per-run VRAM + utilization |
| `torch.cuda.max_memory_allocated()` | in-process peak for batch sizing |
| nsys (`module load nvhpc/25.1`) | CPU↔GPU bottleneck, kernel timeline |
| ncu (after nsys) | per-kernel roofline — 10–100× slower, low priority |

```bash
# nsys on a single row
module load nvhpc/25.1
nsys profile --pytorch=autograd-shapes-nvtx -t cuda,nvtx,osrt,cudnn,cublas \
  -o /fs/scratch/PAS1266/profiles/run \
  python -m graphids exec --row "$(jq -c '.rows[0]' plan.json)"
nsys stats run.nsys-rep
```
