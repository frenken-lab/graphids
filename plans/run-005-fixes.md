# Run 005 Fix Plan

Status: Fix 1 DONE, Fix 2 DONE, Fix 3 RESOLVED (not a bug)
Full post-mortem: `ablation_and_main_005.md` (project root)
Date: 2026-03-31

## Context

Run 005 launched ablation (set_01, 18 configs) + main_results (6 datasets, 2 configs) on Ascend A100s.
22 of 36 SLURM jobs completed. Multiprocess executor works — parallel submission confirmed.
Three categories of failure; two are blocking, one is observability.

## Fix 1: Fusion checkpoint path wiring [DONE]

**Bug:** `pipeline.yaml:61` hardcoded fusion GAT dep as `stage: curriculum`. Configs using `normal` instead
got no GAT checkpoint wired → `gat_ckpt_path=""` → `Path("")` → CWD → `IsADirectoryError`.

**Fix applied (principled — Option A):**
- `pipeline.yaml:59-62` — added `{model: gat, stage: normal}` to fusion depends_on (topology declares all valid paths)
- `component.py:253-262` — `seen_models` dedup: first resolving dep per model wins (curriculum OR normal, never both)
- `datamodule.py:401-404` — guard: raises `ValueError` if vgae/gat ckpt path is empty (defense-in-depth)
- `test_pure.py` — `mini_pipeline` fixture updated, new assertion: normal-only fusion has `gat_ckpt_path` wired

**Verified:**
- 47/47 orchestrate pure tests pass
- `validate-recipe` passes for both `ablation.yaml` and `main_results.yaml`
- All 15 ablation fusion assets have `gat_wired=True` (verified via enumerate_assets dry run)
- main_results fusion assets (2) also wired correctly (these use curriculum, weren't broken, but confirmed)

## Fix 2: Large autoencoder OOM on set_03/set_04 [DONE]

**Bug:** CPU RAM OOM during PyG `InMemoryDataset.process()` — dies in 4 min before training starts.
48G and 67G both insufficient. Not GPU VRAM.

**Root cause:** set_03/set_04 had no v7.0.0 `.pt` cache (set_01/set_02 did), so they were the only datasets
running `process()`. The processing pipeline held 3 full copies in memory simultaneously:
1. `sliding_window_graphs()` builds bulk tensors (all node features, edges concatenated)
2. Loop clones slices into individual `Data` objects → 2nd copy
3. `InMemoryDataset.collate(data_list)` concatenates them back into bulk tensors → 3rd copy

Peak memory: ~3x final tensor size. For 60M+ row datasets this exceeds 67G.

**Fix applied (root cause — zero-copy collation):**
- `features.py:sliding_window_graphs()` — returns `(Data, slices_dict, num_graphs)` directly from bulk
  tensors. No `.clone()` loop, no `list[Data]`, no `collate()` call. The bulk tensors ARE the collated
  format — RLE boundaries become the slices dict. Peak memory: ~1x final tensor size.
- `can_bus.py:process()` — saves `[data, slices]` directly via `atomic_save`. Deleted `save()` override
  (was the collate bottleneck).
- `can_bus.py:_build_graphs()` — returns `(Data, slices, num_arb_ids, num_graphs)`
- `can_bus.py:_write_cache_metadata()` — computes stats from slices diffs instead of iterating data_list
- `test_features.py` — `_get_graph()` helper extracts individual graphs from collated format for assertions

**Verified:**
- 12/12 preprocessing tests pass, 47/47 orchestrate tests pass (59 total)
- Output format is byte-identical to `InMemoryDataset.collate()` output — existing v7.0.0 cache loads fine
- Confirmed via inspecting set_01 cache: same `(Data, slices)` structure, same dtypes, same cumsum pattern

**Already applied (prior session):** KD wall times increased (+2h each) in `resources.yaml`.

## Fix 3: GPU metrics collection [RESOLVED — not broken]

Both original diagnoses were wrong. Investigated 2026-03-31.

**Bug 1 (DeviceStatsMonitor → CSVLogger): NOT MISSING — sparse rows.**
DeviceStatsMonitor calls `logger.log_metrics()` directly (not `self.log()`). CSVLogger's
`ExperimentWriter.log_metrics()` appends each call as a separate dict to `self.metrics` list.
Model metrics and device stats arrive as separate `log_metrics()` calls with the same step →
separate CSV rows, each missing the other's columns.

Evidence: `set_02/vgae_large_autoencoder_bf355e79` metrics.csv has **248 columns** (244 GPU +
model metrics), **930 rows**: 510 with model metrics, 420 with device stats, **0 with both**.
The GPU data IS there — just needs a step-based join to merge sparse rows.

To consume: `pd.read_csv(...).groupby('step').first()` merges the sparse rows.

**Bug 2 (wandb pynvml on Ascend): FULLY WORKING.**
wandb API query across 26 finished Ascend runs shows full GPU metrics:
`system.gpu.0.gpu` (utilization), `system.gpu.0.memory`, `system.gpu.0.enforcedPowerLimitWatts`.
Mean GPU util 55-80% across runs, memory 15-70%. All populated with real values.

The earlier diagnosis that only `correctedMemoryErrors` was present was likely from querying a
short-lived failed run (fusion jobs ran <1 min) where wandb hadn't collected enough samples.

**No code changes needed.** GPU observability is working on both CSVLogger and wandb.

## Changes Made This Session

1. `graphids/orchestrate/component.py` — added `multiprocess_executor` to `build_defs()` for parallel asset materialization
2. `graphids/orchestrate/run.py` — rewrote to read datasets/seeds from recipe, launch all partitions in one call
3. `graphids/config/recipes/main_results.yaml` — new recipe: large teacher + small KD, all 6 datasets
4. `graphids/config/resources.yaml` — KD wall times: vgae.small.autoencoder 2h→4h, gat.small.normal/curriculum 2.5h→4.5h
5. `graphids/orchestrate/profiler.py` + `slurm.py` — added `--cluster`/`-M` flag for cross-cluster sacct queries

## Relaunch Sequence

All 3 fixes resolved. Ready to relaunch.

**IMPORTANT:** set_03/set_04 v7.0.0 cache doesn't exist yet. First relaunch will run `process()`
with the new zero-copy code path. The existing set_01/set_02 caches are format-compatible and
will load fine (cache hit → skip).

```bash
# From Ascend login node (ssh ascend.osc.edu)
python -m graphids run --recipe graphids/config/recipes/main_results.yaml
python -m graphids run --recipe graphids/config/recipes/ablation.yaml
```

Skip logic will skip the 22 already-completed jobs. Only failures rerun.

**Note on metrics.csv:** GPU stats are on separate rows from model metrics. To merge:
`pd.read_csv('metrics.csv').groupby('step').first()`
