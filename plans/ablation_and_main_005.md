# Run 005 — Ablation + Main Results (2026-03-31)

Cluster: Ascend (A100-PCIE-40GB), partition: nextgen
Recipes: `main_results.yaml` (6 datasets × 2 configs) + `ablation.yaml` (set_01 × 18 configs)

## Job Summary

### Main Results (`main_results.yaml`)

| Dataset | large autoencoder | large curriculum | small KD autoencoder | small KD curriculum | large fusion | small KD fusion |
|---------|-------------------|------------------|----------------------|---------------------|--------------|-----------------|
| hcrl_ch | COMPLETED (17m) | COMPLETED (35m) | COMPLETED (14m) | — | — | — |
| hcrl_sa | COMPLETED (13m) | COMPLETED (7m) | COMPLETED (10m) | — | — | — |
| set_01 | COMPLETED (prior) | COMPLETED (3h08m) | COMPLETED (1h39m) | — | — | — |
| set_02 | COMPLETED (3h05m) | RUNNING (3h10m+) | FAILED (exit 10, USR1 wall timeout at 1h55m) | — | — | — |
| set_03 | OOM ×3 (exhausted retries) | blocked | blocked | blocked | blocked | blocked |
| set_04 | OOM ×3 (exhausted retries) | blocked | blocked | blocked | blocked | blocked |

### Ablation (`ablation.yaml`, set_01 only)

| Stage | Status |
|-------|--------|
| Autoencoders (all variants) | COMPLETED |
| Normals (3 conv variants) | COMPLETED |
| Curricula (6 loss×curriculum variants) | COMPLETED |
| Fusions (5 jobs) | ALL FAILED — exit 1 in <1 min |

## Failures

### 1. Fusion: `IsADirectoryError` — checkpoint path wiring bug (ALL 5 fusion jobs)

```
IsADirectoryError: [Errno 21] Is a directory: '/users/PAS2022/rf15/KD-GAT'
```

**Location:** `datamodule.py:402` — `FusionDataModule.setup()` calls `load_inner_model("gat", Path(hp.gat_ckpt_path), device)` but `gat_ckpt_path` contains the project root instead of a checkpoint file path.

**Root cause:** `pipeline.yaml:61` hardcodes fusion's GAT dependency as `stage: curriculum`. In `enumerate_assets()` Pass 2 (`component.py:255-258`), the dependency lookup does `stage_map.get("curriculum")`. For recipe configs that use `normal` instead of `curriculum` (e.g. `ce_normal`, `focal_normal`, `wce_normal`, `unsup_dgi`), `stage_map` has no `"curriculum"` key. `dep_asset` is `None`, the `if dep_asset:` guard silently skips it, and the fusion `StageConfig` ends up with empty `upstream_ckpt_flags` for GAT. `build_cli_args()` never emits `--data.init_args.gat_ckpt_path`, so `FusionDataModule.__init__` gets the default `gat_ckpt_path=""`. `Path("")` resolves to CWD at runtime → `IsADirectoryError`.

**Full chain:**
1. `pipeline.yaml:61` — `depends_on: [{model: gat, stage: curriculum}]` — no `normal` alternative
2. `component.py:257` — `stage_map.get("curriculum")` → `None` for normal-only configs
3. `component.py:258` — `if dep_asset:` guard silently drops the dependency
4. `component.py:315-318` — `build_cli_args` never emits `--data.init_args.gat_ckpt_path`
5. `datamodule.py:332` — default `gat_ckpt_path: str = ""`, no fusion YAML sets it
6. `datamodule.py:402` — `Path("")` → CWD → `IsADirectoryError`

**Affected configs:** `ce_normal`, `focal_normal`, `wce_normal`, `unsup_dgi` (4 of 18 ablation). BUT also affects main_results fusion because the same wiring path is used — the main_results configs use `curriculum`, so the lookup succeeds, but the IOManager `load_input()` may still hand back a wrong path. Need to verify main_results fusion separately.

**Affected jobs:** 4504703, 4504705, 4504706, 4504708, 4505894

### 2. Large autoencoder OOM on set_03, set_04 — CPU RAM, not GPU VRAM

```
Detected 5 oom_kill events in StepId=4498033.batch
```

**Root cause:** CPU RAM OOM (SLURM cgroup kill), NOT GPU VRAM. Jobs die during `Processing...` — PyG's preprocessing/data loading phase — within 4 minutes, before training starts. This is the PyG `InMemoryDataset.process()` call loading raw CSVs into memory.

**Evidence from profiler:**
- All 6 OOM jobs hit 100% memory efficiency (RSS = requested allocation) instantly
- 48G → OOM in ~4 min. 67G (retry 1) → OOM in ~4 min. Pattern is consistent.
- CPU% is 32-45% during OOM (higher than training jobs at 8-24%) — consistent with multi-worker data loading saturating RAM
- set_02 COMPLETED at 48G in 3h05m — set_03/set_04 must be significantly larger

**Retry history:**
| Job | Dataset | Mem | Elapsed | State |
|-----|---------|-----|---------|-------|
| 4498033 | set_03 | 48G | 4:04 | OOM |
| 4498368 | set_03 | 67G | 3:54 | OOM |
| 4498806 | set_03 | 67G | 4:20 | OOM |
| 4498036 | set_04 | 48G | 4:21 | OOM |
| 4498369 | set_04 | 67G | 4:08 | OOM |
| 4498807 | set_04 | 67G | 4:34 | OOM |

Note: adaptive retry scales 48G × 1.4 = 67G, but only got 2 retries (max_retries=2 for OOM). Third retry would be 94G but was never attempted. The issue is 67G still isn't enough — the preprocessing spike for these datasets exceeds it.

**Affected:** set_03, set_04 — all downstream stages blocked.

**Jobs:** 4498033, 4498036, 4498368, 4498369, 4498806, 4498807

### 3. Small KD autoencoder timeout on set_02

**Root cause:** `autoencoder_8e6b9f70_kd_set_02_s42` hit wall time limit (exit code 0:10 = SIGUSR1 from `--signal=B:USR1@300`). Ran for 1h55m on a 2h allocation. set_02 is large enough that small KD autoencoder needs more time.

**Affected:** set_02 KD pipeline — KD curriculum and KD fusion blocked.

**Jobs:** 4509941

## What Worked

- **Multiprocess executor:** Independent assets launched in parallel correctly. Both autoencoders (large + small KD) submitted simultaneously for each dataset.
- **Skip logic:** set_01 large autoencoder had a prior `.complete` marker — curriculum and KD autoencoder started immediately without re-training.
- **Ablation set_01:** All autoencoder, normal, and curriculum variants completed. Only fusion stage is broken.
- **Small datasets (hcrl_ch, hcrl_sa):** Full main_results pipeline completed through curriculum for both configs. Fast (7–35min).

## Exact Fixes

### Fix 1: Fusion checkpoint path wiring [BLOCKING]

**Option A (recommended) — declare both GAT stages in pipeline.yaml + deduplicate in code:**

`pipeline.yaml:59-62` — add `normal` as alternative GAT source:
```yaml
  fusion:
    depends_on:
      - { model: vgae, stage: autoencoder }
      - { model: gat, stage: curriculum }
      - { model: gat, stage: normal }
```

`component.py:253-262` — deduplicate so only the first resolving GAT dep is wired:
```python
seen_models: set[str] = set()
for dep in stage_def.get("depends_on", []):
    dep_stage = dep["stage"]
    dep_model = dep["model"]
    dep_asset = stage_map.get(dep_stage)
    if not dep_asset:
        continue
    if dep_model in seen_models:
        continue
    seen_models.add(dep_model)
    upstream_names.append(dep_asset)
    flag = _CKPT_FLAG.get(dep_model, "")
    if flag:
        upstream_flags[dep_asset] = flag
```

**Option B (minimal patch) — fallback in enumerate_assets only:**

`component.py:257` — add curriculum→normal fallback:
```python
dep_stage = dep["stage"]
if dep_stage == "curriculum" and dep_stage not in stage_map:
    dep_stage = "normal"
```

**Prevention — guard empty ckpt paths at definition time:**

After the `upstream_flags` loop in `enumerate_assets()`, add:
```python
required_deps = {dep["model"] for dep in stage_def.get("depends_on", [])}
resolved_deps = set()
for name in upstream_names:
    if name in upstream_flags:
        # infer model from flag
        for model, flag in _CKPT_FLAG.items():
            if upstream_flags[name] == flag:
                resolved_deps.add(model)
missing = required_deps - resolved_deps - {"preprocess"}
if missing:
    raise ValueError(
        f"Asset '{asset_name}': unresolved checkpoint deps {missing}. "
        f"stage_map={stage_map}, depends_on={stage_def['depends_on']}"
    )
```

Also add a guard in `FusionDataModule.setup()` (`datamodule.py:402`):
```python
if not hp.gat_ckpt_path:
    raise ValueError("gat_ckpt_path is empty — upstream GAT checkpoint not wired")
```

### Fix 2: Large autoencoder OOM on set_03/set_04 [BLOCKING]

This is a CPU RAM spike during PyG `InMemoryDataset.process()`, not a training-time issue. 67G wasn't enough, so simply bumping `mem` may not be sufficient — the preprocessing loads all raw CSVs into memory at once.

**Option A — brute force memory increase:**
`resources.yaml:59-66`:
```yaml
  large:
    autoencoder:
      mode: gpu_train
      time: "06:00:00"    # was 04:00:00
      mem: "128G"          # was 48G — must clear preprocessing spike
      cpus_per_task: 4
      num_workers: 2       # was 3 — reduce RSS from worker copies
```
Risk: Ascend A100 nodes have ~362G RAM but 128G is a big ask. May wait longer in queue.

**Option B — reduce preprocessing memory footprint:**
Investigate `InMemoryDataset.process()` in the datamodule — if it loads all CSVs into a single list before writing `.pt`, chunking the processing would fix the root cause. This is the real fix but requires code changes.

**Option C — increase max_retries to 3:**
`resources.yaml` failure_reactions: change `max_retries: 2` → `3` for OOM. Third retry would try 94G (67 × 1.4), which might be enough. Cheapest change but not guaranteed.

### Fix 3: KD autoencoder wall time on set_02 [APPLIED]

**Already fixed** — `resources.yaml` updated this session:

| Profile | Before | After |
|---|---|---|
| `vgae.small.autoencoder` | `02:00:00` | `04:00:00` |
| `gat.small.normal` | `02:30:00` | `04:30:00` |
| `gat.small.curriculum` | `02:30:00` | `04:30:00` |

Note: exit code 10 (SIGUSR1) maps to SLURM state `FAILED`, not `TIMEOUT`. The adaptive retry only handles `OUT_OF_MEMORY` and `TIMEOUT`. Wall-time-killed jobs won't auto-retry. The increased allocations should prevent this from recurring.

## Profiler Output

Run: `python -m graphids profile --since 2026-03-31 --cluster ascend`

```
=== Job Summary ===

JobID        State         Elapsed    RSS ReqMem  Mem%  CPU% Stage
------------------------------------------------------------------
4498032      COMPLETED    00:17:26 48.0G   48G  100%   14%  hcrl_ch/autoencoder
4498033      OUT_OF_MEMORY 00:04:04 48.0G   48G  100%   44%  set_03/autoencoder
4498034      COMPLETED    00:12:52 48.0G   48G  100%    8%  hcrl_sa/autoencoder
4498035      COMPLETED    03:05:13 48.0G   48G  100%   14%  set_02/autoencoder
4498036      OUT_OF_MEMORY 00:04:21 48.0G   48G  100%   45%  set_04/autoencoder
4498037      COMPLETED    03:08:14 36.0G   36G  100%   14%  set_01/curriculum
4498038      COMPLETED    01:39:24 36.0G   36G  100%   24%  set_01/autoencoder
4498233      COMPLETED    00:34:19 36.0G   36G  100%   23%  set_01/autoencoder
4498234      COMPLETED    00:38:13 36.0G   36G  100%   22%  set_01/normal
4498235      COMPLETED    01:20:56 36.0G   36G  100%   23%  set_01/normal
4498236      COMPLETED    01:16:24 36.0G   36G  100%   22%  set_01/normal
4498237      COMPLETED    01:31:00 36.0G   36G  100%   24%  set_01/autoencoder
4498238      COMPLETED    01:35:06 36.0G   36G  100%   24%  set_01/autoencoder
4498239      COMPLETED    01:52:31 36.0G   36G  100%   22%  set_01/curriculum
4498240      COMPLETED    03:06:11 36.0G   36G  100%   14%  set_01/curriculum
4498241      COMPLETED    01:35:48 36.0G   36G  100%   24%  set_01/autoencoder
4498368      OUT_OF_MEMORY 00:03:54 67.0G   67G  100%   38%  set_03/autoencoder
4498369      OUT_OF_MEMORY 00:04:08 67.0G   67G  100%   39%  set_04/autoencoder
4498806      OUT_OF_MEMORY 00:04:20 67.0G   67G  100%   32%  set_03/autoencoder
4498807      OUT_OF_MEMORY 00:04:34 67.0G   67G  100%   37%  set_04/autoencoder
4499070      COMPLETED    00:09:48 36.0G   36G  100%   10%  hcrl_sa/autoencoder
4499071      COMPLETED    00:06:30 36.0G   36G  100%   16%  hcrl_sa/curriculum
4499213      COMPLETED    00:14:08 36.0G   36G  100%   18%  hcrl_ch/autoencoder
4499214      COMPLETED    00:34:49 36.0G   36G  100%   22%  hcrl_ch/curriculum
4500667      COMPLETED    02:24:11 36.0G   36G  100%   14%  set_01/curriculum
4504348      COMPLETED    01:55:17 36.0G   36G  100%   17%  set_01/curriculum
4504702      COMPLETED    01:58:46 36.0G   36G  100%   17%  set_01/curriculum
4504703      FAILED       00:00:47 16.0G   16G  100%   15%  set_01/fusion
4504704      COMPLETED    01:59:27 36.0G   36G  100%   17%  set_01/curriculum
4504705      FAILED       00:00:57 16.0G   16G  100%   13%  set_01/fusion
4504706      FAILED       00:00:57 16.0G   16G  100%   13%  set_01/fusion
4504707      COMPLETED    01:59:22 36.0G   36G  100%   17%  set_01/curriculum
4504708      FAILED       00:00:38 16.0G   16G  100%   15%  set_01/fusion
4505894      FAILED       00:00:53 16.0G   16G  100%   11%  set_01/fusion
4509940      COMPLETED    03:25:34 36.0G   36G  100%   14%  set_02/curriculum
4509941      FAILED       01:54:59 36.0G   36G  100%   24%  set_02/autoencoder
```

Totals: 22 COMPLETED, 6 OUT_OF_MEMORY, 7 FAILED, 1 RUNNING

## GPU Metrics — BROKEN on Ascend

Two independent failures mean we have NO GPU utilization data from this run:

### 1. DeviceStatsMonitor → CSVLogger: not writing

`DeviceStatsMonitor` is configured in `trainer.yaml:22` and calls `logger.log_metrics()` directly.
But `metrics.csv` only contains model metrics (`epoch, step, train_acc, train_loss, val_acc, val_loss`).
No CUDA allocator columns (`allocated_bytes`, `reserved_bytes`, etc.) appear.

**Evidence:** `head -1 .../lightning_logs/version_0/metrics.csv` → 6 columns, zero GPU columns.

**Likely cause:** `_get_and_log_device_stats` gates on `trainer._logger_connector.should_update_logs`,
which only returns True at `log_every_n_steps` intervals. DeviceStatsMonitor fires on `on_train_batch_start`
but the logger connector may not be ready yet. Or CSVLogger's `log_metrics` doesn't merge these rows
with the model's `self.log()` rows — they may be silently dropped or written as empty rows.

### 2. wandb pynvml GPU stats: broken on Ascend

Older Pitzer runs (e.g. `clean-cherry-100`) have full system metrics:
`system.gpu.0.gpu`, `system.gpu.0.memory`, `system.gpu.0.temp`, `system.gpu.0.powerWatts`

Today's Ascend runs only have: `system.gpu.0.correctedMemoryErrors` — no util, no memory, no temp, no power.

**Evidence:** wandb API `run.history(stream='events')` for all 8 Ascend runs shows identical truncated keys.
`wandb-metadata.json` correctly detects `NVIDIA A100-PCIE-40GB`, so the GPU is visible.

**Likely cause:** pynvml on Ascend can't read the A100 performance counters — either the NVIDIA driver
version on Ascend nodes doesn't expose them via NVML, or the SLURM cgroup restricts access.
Only error counters (correctedMemoryErrors) are readable.

### Impact

- Cannot analyze GPU utilization, VRAM usage over time, or power draw for any Run 005 job
- Cannot right-size GPU allocations or detect underutilization
- The profiler (`python -m graphids profile`) only has SLURM sacct data (CPU RSS, wall time)

### Fixes Needed

1. **DeviceStatsMonitor → CSV**: Test locally whether CSVLogger actually receives the metrics.
   If Lightning bug, workaround: add a custom callback that calls `self.log()` with device stats
   (which CSVLogger does capture).
2. **wandb pynvml on Ascend**: Check `nvidia-smi` on an Ascend node to verify driver supports
   utilization queries. If it does, file wandb issue. If not, add explicit `nvidia-smi --query-gpu`
   polling in `_preamble.sh` or a custom callback.
