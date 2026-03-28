# Priority Tier List + Implementation Details

> Status: **active** | Date: 2026-03-27

## Tier 1: Unblocks running experiments (this week)

### 1.1 Restore fast collation (~60 lines)

**Impact:** GPU util 30% → 82% on VGAE. Every GPU job runs 2.7× faster. Directly
reduces allocation burn.

**Root cause:** `_FastCollate` was added in `527857b` (2026-03-25), measured at T_c=25ms
(vs 70ms standard), then deleted same day in `7ece283` during PyG API cleanup.

**What it does:** Instead of `separate()` → N Data objects → `Batch.from_data_list()`
(3600 Python ops per batch), it slices directly from `InMemoryDataset._data` tensors
using vectorized `_range_index` (6 C++ kernel launches per batch).

**File:** `graphids/core/preprocessing/datamodule.py`

**Implementation — restore as `collate_fn` in `make_graph_loader`:**

The original had 3 classes (`_FastCollate`, `_SlicesBatchSampler`, `_IndexDataset` = 190
lines). We only need the collation function (~60 lines) because `DynamicBatchSampler`
already handles batch composition.

The key issue: `DynamicBatchSampler` yields lists of indices, but standard PyG `DataLoader`
calls `dataset[i]` for each index (triggers `separate()`) then `Batch.from_data_list()`.
We need to intercept this and slice directly.

**Approach:** `_IndexDataset` wrapper + `fast_collate` function, passed to a plain
`torch.utils.data.DataLoader` (not PyG's DataLoader, which forces `from_data_list`).

**Changes to `datamodule.py`:**

1. Add `_IndexDataset` class (~15 lines) — wraps dataset, returns physical indices
   instead of Data objects. Resolves `_indices` mapping for train/val splits.

2. Add `_make_fast_collate_fn(dataset)` (~45 lines) — closes over `dataset._data` and
   `dataset.slices`. Returns a `fast_collate(indices) -> Batch` function that:
   - Computes node/edge slice boundaries from `slices` tensors (vectorized)
   - Builds flat gather index via `_range_index` (vectorized, 0 Python loops)
   - Gathers x, edge_index, edge_attr, node_id via single `index_select` per attr
   - Applies edge_index offsets (vectorized cumsum)
   - Builds batch vector and ptr (vectorized `repeat_interleave`)

3. Modify `make_graph_loader` (~10 lines) — when dataset is `InMemoryDataset` with
   `_data` tensors, use `_IndexDataset` + `fast_collate` + plain `DataLoader`.
   Otherwise fall through to PyG `DataLoader`.

**Exact insertion point:** `datamodule.py:31-47` (current `make_graph_loader`).
Replace with the version from `527857b` adapted to current code.

**Source code:** `git show 527857b:graphids/core/preprocessing/datamodule.py` lines 30-160
has the exact implementation. Port it forward, dropping `_SlicesBatchSampler` (replaced
by `DynamicBatchSampler`).

**Callers that automatically benefit (no changes needed):**
- `CANBusDataModule._build_loader` (`datamodule.py:184`) — already calls `make_graph_loader`
- `CurriculumDataModule.setup` (`curriculum.py:162`) — already calls `make_graph_loader`
- `CurriculumDataModule.val_dataloader` (`curriculum.py:182`) — already calls `make_graph_loader`

**Callers that won't use fast path (plain `list[Data]`, no `_data` tensors):**
- `curriculum.py:182` `val_dataloader` when `self.val_data` is a plain list — falls
  through to PyG DataLoader. This is fine (val is not the hot loop).

**Verification:**
```bash
# 5-epoch VGAE on set_02 with 2 workers — expect ~80% GPU util
sbatch --partition=gpu --gres=gpu:1 --time=00:30:00 --mem=36G \
  --cpus-per-task=4 --account=PAS1266 \
  --output=slurm_logs/verify_fast_collate_%j.out \
  --wrap="source scripts/slurm/_preamble.sh && \
    srun python -m graphids fit \
      --model graphids.core.models.vgae.VGAEModule \
      --data graphids.core.preprocessing.datamodule.CANBusDataModule \
      --data.dataset=set_02 --data.num_workers=2 \
      --trainer.max_epochs=5 --trainer.accelerator=gpu --trainer.devices=1 \
      --trainer.callbacks+=pytorch_lightning.callbacks.DeviceStatsMonitor"
```
If GPU util is ~80%, fast collation is working. If ~30%, something went wrong.

---

### 1.2 Write `resources.yaml` (~50 lines)

**Impact:** Stops OOM failures (103 jobs in March). Provides the data for the
orchestrator's adaptive retry.

**File:** `graphids/config/defaults/resources.yaml` (new file)

**Implementation:**

```yaml
# Resource profiles — single source of truth for SLURM allocation + DataLoader config.
# RAM formula: 15G + num_workers × dataset_size + 4G overhead
# T_collate ≈ 25ms with fast collation restored (tier 1.1)
# Workers: VGAE=3 (saturates V100), GAT=2 (saturates V100)

resource_profiles:
  vgae:
    medium:
      autoencoder:
        partition: gpu
        gres: "gpu:1"
        time: "02:30:00"
        mem: "37G"              # 15 + 3×5.9 + 4
        cpus_per_task: 4
        num_workers: 3
      curriculum:
        partition: gpu
        gres: "gpu:1"
        time: "02:30:00"
        mem: "37G"
        cpus_per_task: 4
        num_workers: 3
    large:
      autoencoder:
        partition: gpu
        gres: "gpu:1"
        time: "04:00:00"
        mem: "48G"
        cpus_per_task: 4
        num_workers: 3

  gat:
    medium:
      normal:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"
        mem: "28G"              # 15 + 2×5.9 + 4
        cpus_per_task: 3
        num_workers: 2
      curriculum:
        partition: gpu
        gres: "gpu:1"
        time: "03:00:00"
        mem: "28G"
        cpus_per_task: 3
        num_workers: 2
    large:
      normal:
        partition: gpu
        gres: "gpu:1"
        time: "05:00:00"
        mem: "37G"
        cpus_per_task: 4
        num_workers: 3

  dqn:
    medium:
      fusion:
        partition: gpu
        gres: "gpu:1"
        time: "01:00:00"
        mem: "16G"
        cpus_per_task: 2
        num_workers: 0

  bandit:
    medium:
      fusion:
        partition: gpu
        gres: "gpu:1"
        time: "01:00:00"
        mem: "16G"
        cpus_per_task: 2
        num_workers: 0

  preprocess:
    any:
      preprocess:
        partition: cpu
        time: "02:00:00"
        mem: "72G"
        cpus_per_task: 8
        num_workers: 0

  test:
    any:
      test:
        partition: cpu
        time: "00:30:00"
        mem: "16G"
        cpus_per_task: 8
        num_workers: 0

# Adaptive retry — orchestrator scales resources on failure
failure_reactions:
  OUT_OF_MEMORY:
    scale_mem: 1.4
    max_retries: 2
  TIMEOUT:
    scale_time: 1.5
    max_retries: 1
  NODE_FAIL:
    max_retries: 2
```

**No code changes needed** — this is a data file. The orchestrator (tier 2) reads it.
Until then, it serves as documentation for manual sbatch submissions.

---

### 1.3 Fix CurriculumDataModule worker restart (~10 lines)

**Impact:** Eliminates ~80 min overhead per 300-epoch curriculum run
(300 epochs × 3 workers × ~4s spawn startup = 60-80 min).

**Root cause:** `CurriculumDataModule.train_dataloader()` returns a cached `_train_loader`
but `setup()` creates it with `persistent_workers` based on `make_graph_loader` defaults.
The issue is that `val_dataloader()` creates a **new** DataLoader every call
(`curriculum.py:171-184`), and the manual `_current_epoch` counter increments incorrectly
when Lightning calls `train_dataloader()` multiple times (sanity check).

**File:** `graphids/core/preprocessing/curriculum.py`

**Changes:**

1. **Replace manual epoch counter with `self.trainer.current_epoch`** (`curriculum.py:167-168`):

```python
# Before (curriculum.py:166-169):
def train_dataloader(self):
    self._batch_sampler.set_epoch(self._current_epoch)
    self._current_epoch += 1
    return self._train_loader

# After:
def train_dataloader(self):
    self._batch_sampler.set_epoch(self.trainer.current_epoch)
    return self._train_loader
```

Delete `self._current_epoch = 0` from `__init__` (`curriculum.py:111`).

2. **Cache val_dataloader** — build once in `setup()`, return cached (`curriculum.py:171-184`):

```python
# Before (curriculum.py:171-184): builds new DBS + new DataLoader every call
def val_dataloader(self):
    hp = self.hparams
    bs = max(8, hp.batch_size)
    if hp.dynamic_batching:
        info = compute_node_budget(bs, hp, conv_type=hp.conv_type, heads=hp.heads)
        num_steps = max(1, len(self.val_data) * 30 // info.budget)
        sampler = DynamicBatchSampler(...)
        return make_graph_loader(self.val_data, batch_sampler=sampler, ...)
    return make_graph_loader(self.val_data, batch_size=bs, ...)

# After: build once in setup(), cache as self._val_loader
# Add to setup() after line 164:
    self._val_loader = self._build_val_loader()

def val_dataloader(self):
    return self._val_loader
```

**Verification:** Run a short curriculum job (5 epochs). Check that epoch transitions
are instantaneous (no 3-5s worker spawn gap visible in logs).

---

### 1.4 Wire `SLURMEnvironment(auto_requeue=true)` into training configs

**Impact:** Preemption (TIMEOUT, wall-time signal) becomes a non-event.
Lightning saves checkpoint + SLURM requeues + Lightning resumes automatically.

**Verified:** Spike job `46012629` confirmed `SLURMEnvironment` + `ModelCheckpoint` works
on Pitzer V100 nodes.

**What needs to happen:** Add to the YAML config template used for all training stages.
No Python code changes.

**Example stage YAML addition:**

```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        save_top_k: 1
        save_last: true
        mode: min
        filename: "best_model"
    - class_path: pytorch_lightning.callbacks.DeviceStatsMonitor
  plugins:
    - class_path: pytorch_lightning.plugins.environments.SLURMEnvironment
      init_args:
        auto_requeue: true
```

**SLURM script requirement:** `#SBATCH --signal=B:USR1@300` (sends USR1 5 min before
wall time). Already in `_preamble.sh:65` as a trap — with `SLURMEnvironment`, the shell
trap becomes redundant (Lightning catches the signal internally).

**Verification:** Submit a training job with `--time=00:10:00` on `gpudebug`, let it
hit wall time. Check:
- `last.ckpt` was written before termination
- Job was requeued (`squeue` shows same job ID in PENDING)
- On restart, training resumes from the checkpoint (log shows "Restoring states from...")

---

## Tier 2: Stops the resubmit-and-pray cycle (next week)

| # | What | File | Lines | Depends on |
|---|---|---|---|---|
| 2.1 | Orchestrator: `run_pipeline`, `submit_one`, `poll_and_retry` | `graphids/orchestrate/submit.py` | +100 | 1.2 (resources.yaml) |
| 2.2 | DAG topology: `build_dag_topology`, `topo_sort` | `graphids/orchestrate/dag.py` | +80 | pipeline.yaml (exists) |
| 2.3 | Resource loader: `get_resources`, `scale_resources` | `graphids/orchestrate/resources.py` | +60 | 1.2 (resources.yaml) |
| 2.4 | Config generator: ablation spec → stage YAMLs | `graphids/orchestrate/generate_configs.py` | +50 | 1.2, 1.4 |
| 2.5 | Trim `_preamble.sh` / `_epilog.sh` | `scripts/slurm/` | -110 | 1.4 (SLURMEnvironment) |
| 2.6 | Delete `cluster.py` | `graphids/cluster.py` | -74 | 2.1 |

Details: see `plans/pipeline-consolidation.md`.

## Tier 3: Code consolidation (when you have a clean week)

| # | What | File(s) | Lines | Details in |
|---|---|---|---|---|
| 3.1 | Dead code deletion | models + preprocessing | -200 | models-consolidation.md §4, preprocessing §1 |
| 3.2 | `GraphModuleBase` shared base | `models/_training.py` | net -90 | models-consolidation.md §2, §3 |
| 3.3 | Delete `configure_optimizers` + wire CLI | `__main__.py` + 3 modules | net -25 | models-consolidation.md §1 |
| 3.4 | Preprocessing DataModule conventions | 3 DataModules | net +31 | preprocessing-consolidation.md §6 |
| 3.5 | Dissolve `registry.py` | `models/` | net -73 | models-consolidation.md §6 |
| 3.6 | Inline `_training.py` single-use utilities | `models/` | net -18 | models-consolidation.md §7 |
| 3.7 | `temporal.py` checkpoint fix | `models/temporal.py` | -10 | models-consolidation.md §8 |

## Tier 4: When writing the paper

| # | What | File(s) | Lines | Details in |
|---|---|---|---|---|
| 4.1 | Artifacts rewrite (embeddings, CKA, loss landscape) | `core/artifacts/` | net -220 | artifacts-consolidation.md |
| 4.2 | DQN/Bandit → LightningModules | `models/` | net -120 | models-consolidation.md §5 |
| 4.3 | Memory bloat spike (prefetch thread) | spike | experimental | memory_profiling/resource_plan §Problem 2 |

## Cross-references

| Plan file | Scope |
|---|---|
| `plans/models-consolidation.md` | 13 model files, -287 lines |
| `plans/preprocessing-consolidation.md` | 8 data files, -179 lines |
| `plans/artifacts-consolidation.md` | 6 artifact files, -220 lines |
| `plans/pipeline-consolidation.md` | Orchestration + SLURM |
| `memory_profiling/resource_plan_2026_03_27.md` | Resource profiles + collation analysis |
