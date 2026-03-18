# Plan: Pipeline Layer Consolidation (v2)

**Created**: 2026-03-17
**Updated**: 2026-03-17 (v2 — integrated ecosystem audit, reworked phases)
**Status**: Draft — awaiting review
**Scope**: `graphids/pipeline/` (5,750 lines across 23 files)

---

## Current Structure & Function Map

### Top-level (`pipeline/`)

| File | Lines | Functions / Classes | Role |
|------|-------|-------------------|------|
| `__init__.py` | 9 | `build_cli_cmd`, `STAGE_FNS` re-exports | Gateway |
| `cli.py` | 709 | `main()`, `_build_parser()`, `_run_single_stage()` (194-line god fn), `_parse_dot_overrides()`, `_run_preprocess()`, `_run_tune()`, `_run_sweep_pipeline()`, `_run_lake()`, `_run_orchestrate()`, `_setup_mlflow()`, `_parse_seeds()` | Entry point, MLflow setup, archive/restore, manifest writing |
| `artifacts.py` | 150 | `get_artifact()`, `put_artifact()`, `artifact_exists()`, `_find_mlflow_run()`, `_id_parts()`, `_artifact_group()`, `_fs_artifact_path()` | Cache-first artifact resolution |
| `serve.py` | 207 | `PredictRequest`, `PredictResponse`, `HealthResponse`, `_load_models()`, `health()`, `predict()` | FastAPI inference server |
| `subprocess_utils.py` | 88 | `build_cli_cmd()` | Shared CLI command builder |
| `validate.py` | 103 | `validate()`, `validate_datasets()`, `_artifact_exists()` | Pre-flight validation |

### Stages (`pipeline/stages/`)

| File | Lines | Functions / Classes | Role |
|------|-------|-------------------|------|
| `__init__.py` | 21 | `STAGE_FNS` dict | Stage registry |
| `training.py` | 189 | `train_autoencoder()`, `train_curriculum()`, `train_normal()`, `_score_difficulty()`, `_save_training_metrics()`, `_resume_ckpt_path()`, `_save_and_cleanup()` | Training entry points |
| `evaluation.py` | 641 | `evaluate()` (305-line god fn), `_run_gat_inference()`, `_run_vgae_inference()`, `_run_fusion_inference()`, `_vgae_threshold()`, `_compute_metrics()`, `_load_test_data()`, `_collect_layer_representations()`, `_save_cka()`, `_linear_cka()` | Multi-model evaluation + artifact export |
| `fusion.py` | 177 | `train_fusion()`, `_train_dqn_fusion()`, `_train_mlp_fusion()`, `_train_weighted_avg_fusion()` | DQN/MLP/WeightedAvg fusion training |
| `temporal.py` | 303 | `TemporalGraphDataset`, `collate_temporal()`, `TemporalLightningModule`, `train_temporal()` | Temporal graph classification |
| `data_loading.py` | 212 | `training_preamble()`, `load_data()`, `make_dataloader()`, `cache_predictions()`, `graph_label()`, `compute_node_budget()`, `_safe_num_workers()`, `_estimate_tensor_count()`, `_estimate_dynamic_steps()` | Dataset loading + DQN state caching |
| `modules.py` | 278 | `VGAEModule`, `GATModule`, `CurriculumDataModule`, `_teacher_to_device()`, `_teacher_offload()`, `_curriculum_sample()` | Lightning modules |
| `trainer_factory.py` | 336 | `prepare_kd()`, `resolve_teacher_path()`, `_load_teacher()`, `make_projection()`, `_extract_state_dict()`, `load_frozen_cfg()`, `load_model()`, `build_optimizer_dict()`, `_setup_mlflow_autolog()`, `make_trainer()` | Trainer + KD lifecycle |
| `batch_sizing.py` | 40 | `effective_batch_size()`, `resolve_batch_config()` | Batch size computation |
| `utils.py` | 61 | `cleanup()` + re-exports from 4 submodules | Convenience layer |

### Orchestration (`pipeline/orchestration/`)

| File | Lines | Functions / Classes | Role |
|------|-------|-------------------|------|
| `__init__.py` | 20 | Lazy gateway (`fire_and_forget`, `build_dagster_assets`) | Gateway |
| `dagster_defs.py` | 476 | `PipesSlurmResource`, `pipeline_partitions`, `_make_stage_asset()`, `_make_hf_push_asset()`, `_make_rebuild_catalog_asset()`, `DagNode`, `build_dag_topology()`, `build_dagster_assets()`, `fire_and_forget()` | Dagster assets + fire-and-forget SLURM |
| `pipes_slurm.py` | 539 | `SlurmJobFailed`, `get_resources()`, `scale_resources()`, `generate_sbatch_script()`, `PipesSlurmClient` | SLURM sbatch/sacct wrapper |
| `sweep_pipeline.py` | 461 | `SweepStep`, `SWEEP_DAG`, `run_sweep_pipeline()`, `_ensure_sweep_run()`, `load_best_config()` | Multi-stage HPO sweep |
| `tune_config.py` | 376 | `run_tune()`, `_build_search_space()`, `_trainable()`, `export_best_config()` | Ray Tune HPO |
| `job.py` | 114 | `ResourceSpec`, `JobState`, `JobSpec` | Pydantic job models |
| `dagster_resources.py` | 59 | `save_retry_state()`, `load_retry_state()`, `clear_retry_state()` | JSON retry state persistence |
| `store.py` | 204 | `PipelineStore` (SQLite state for sweep resume) | Sweep state persistence |

## Dependency Graph

```
cli.py ──→ stages/{training,evaluation,fusion,temporal} (dispatch)
       ──→ artifacts.py (put_artifact)
       ──→ validate.py (pre-flight)
       ──→ orchestration/dagster_defs.py (fire_and_forget)
       ──→ orchestration/sweep_pipeline.py (run_sweep_pipeline)
       ──→ orchestration/tune_config.py (run_tune)

stages/training.py ──→ stages/utils.py ──→ data_loading, batch_sizing, trainer_factory
                   ──→ stages/modules.py (VGAEModule, GATModule, CurriculumDataModule)

stages/evaluation.py ──→ stages/utils.py (load_model, make_dataloader)
                     ──→ artifacts.py (artifact_exists, get_artifact)

stages/fusion.py ──→ stages/utils.py (cache_predictions, load_model, cleanup)
                 ──→ core/models/dqn.py (3 fusion agents)

stages/data_loading.py ──→ core/preprocessing (PreprocessingPipeline, get_batch_index)
                       ──→ core/models/registry (extractors)

orchestration/dagster_defs.py ──→ pipes_slurm.py, subprocess_utils.py, dagster_resources.py
orchestration/sweep_pipeline.py ──→ tune_config.py, pipes_slurm.py, store.py, subprocess_utils.py
```

---

## Exhaustive Ecosystem Audit (2026-03-17)

### submitit (Meta) — SLURM handler

**Verdict: Do not adopt.** Fundamental execution model mismatch.

| Capability | submitit | Current custom | Verdict |
|---|---|---|---|
| sbatch script generation | Generates for pickled callables, not CLI | Custom shell scripts with preamble/epilog | **KEEP** |
| Resource profiles (YAML) | No concept of named profiles | Per-(model,scale,stage) lookup | **KEEP** |
| Adaptive retry (OOM→2× mem) | Only timeout requeue, same resources | OOM/TIMEOUT/NODE_FAIL with scaling | **KEEP** |
| Artifact validation | None | Pydantic contracts post-completion | **KEEP** |
| Dependency chains | Passes `--dependency` flag but no DAG builder | `fire_and_forget()` with topo sort | **PARTIAL** |
| sacct querying | `SlurmInfoWatcher` — well-implemented | Custom `_sacct_query()` | **REPLACE** (~60 lines) |
| Dagster Pipes integration | None | Custom env var passing | **KEEP** |
| Signal handling | USR2 (conflicts with Lightning's USR1) | USR1 via `--signal=B:USR1@180` | **KEEP** |
| Checkpoint resume | Pickle-based (incompatible with Lightning ckpt) | `KD_GAT_CKPT_PATH` env var | **KEEP** |

**Net: saves ~60 lines but breaks signal handling, Dagster Pipes, and checkpoint resume. Not worth it.**

### PyTorch ecosystem — function-level replacement audit

| Custom Code | Lines | Best Candidate | Replaces? | Source |
|---|---|---|---|---|
| `_compute_metrics()` | 60 | `torchmetrics.MetricCollection` | **PARTIAL** (~40 of 60) | [torchmetrics all-metrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html) |
| Detection-at-FPR, Youden's J | 15 | None | **KEEP** | [torchmetrics all-metrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html) (not listed) |
| `CurriculumDataModule` | 60 | None | **KEEP** | [Lightning callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) (no curriculum) |
| Teacher CPU offloading | 15 | `BaseFinetuning` | **KEEP** (handles freeze, not device) | [BaseFinetuning API](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html) |
| VGAE latent KD loss | 28 | `torchdistill` | **KEEP** (CNN-only, no graph) | [PyTorch KD tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) |
| GAT soft-label KD | 18 | `torchdistill` | **KEEP** (8 lines of `F.kl_div`) | [Deprecated KD toolkit](https://github.com/georgian-io/Knowledge-Distillation-Toolkit) |
| `make_trainer()` | 54 | Lightning Trainer | **KEEP** (already idiomatic) | [Trainer docs](https://lightning.ai/docs/pytorch/stable/common/trainer.html) |
| `build_optimizer_dict()` | 32 | Lightning | **KEEP** (already canonical pattern) | Same |
| Layer representations (CKA) | 20 | Captum `LayerActivation` | **KEEP** (awkward with PyG Data) | [Captum Layer API](https://captum.ai/api/layer.html) |
| Attention capture | 10 | PyG `AttentionExplainer` | **KEEP** (aggregates, loses per-layer) | [AttentionExplainer](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.explain.algorithm.AttentionExplainer.html) |
| CKA formula | 10 | `ckatorch` / `torch-cka` | **KEEP** (10 lines vs new dep) | [ckatorch](https://github.com/RistoAle97/centered-kernel-alignment), [torch-cka](https://pypi.org/project/torch-cka/) |
| Single-sample eval loops | 100 | PyG `DataLoader` + `unbatch()` | **PARTIAL** (~15 lines + 10-50× speedup) | [PyG unbatch](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html) |
| `TensorReplayBuffer` | 50 | TorchRL `ReplayBuffer` | **PARTIAL** (saves ~40 but adds ~100MB dep) | [TorchRL ReplayBuffer](https://docs.pytorch.org/rl/stable/reference/generated/torchrl.data.ReplayBuffer.html) |
| `compute_node_budget()` | 20 | `DynamicBatchSampler` | **KEEP** (no auto-estimation) | [DynamicBatchSampler source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dynamic_batch_sampler.html) |
| Artifact resolution | 150 | MLflow native / DVC | **KEEP** (cache→fs→MLflow is the value-add) | [MLflow artifacts API](https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html) |
| MLflow manual logging | 40 | `autolog()` | **KEEP** (autolog misses eval artifacts) | [MLflow autolog source](https://github.com/mlflow/mlflow/blob/master/mlflow/pytorch/_lightning_autolog.py) |
| Gradient checkpointing | 10 | Lightning Trainer | **KEEP** (already using `torch.utils.checkpoint`) | [torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html) |
| Multi-model eval orchestrator | 305 | None | **KEEP** (no GNN eval framework) | [PyG utils](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html) |
| CLI boilerplate | 130 | Typer | **PARTIAL** (saves ~70 lines) | N/A |
| Config overrides (`-O`) | 20 | Hydra/OmegaConf | **KEEP** (conflicts with Pydantic+YAML) | N/A |

### Ecosystem verdict

**3 actionable replacements** (total ~55 lines saved + significant eval speedup):

1. **`torchmetrics.MetricCollection`** → replace `_compute_metrics()` sklearn calls (~40 lines). Already a transitive dep via pytorch-lightning. Doesn't help with detection-at-FPR or Youden's J (~15 lines stay custom).

2. **PyG `DataLoader` + `unbatch()`** → batch eval inference in `_run_gat_inference()` / `_run_vgae_inference()`. ~15 lines saved + **10-50× eval speedup**. Already a dep. This is the highest-value change.

3. **Typer for CLI** → replace `_build_parser()` argparse (~70 lines). Low priority — CLI works, not growing.

**Everything else is irreducibly domain-specific or already delegated to the right library.** No package handles: VGAE latent KD, curriculum difficulty sampling, adaptive SLURM retry, cache-first artifact resolution, fusion reward computation, teacher CPU offloading, or multi-model eval orchestration.

---

## Bugs

### B1: `trainable_label` undefined (`tune_config.py:325`)
```python
tags={"trainable_mode": trainable_label}  # NameError at runtime
```
**Fix**: Replace with string literal `"subprocess"`.

### B2: `partition: str = "serial"` default (`job.py:30`)
Per MEMORY.md, correct OSC partition is `cpu`.
**Fix**: Change default to `"cpu"`.

---

## Hardcoded Values → Config

### H1: VGAE loss component weights (`modules.py:81`, duplicated `training.py:179`)
```python
task_loss = recon + 0.1 * canid + 0.05 * nbr_loss + 0.01 * kl_loss  # modules.py
scores.append(recon + 0.1 * canid)                                    # training.py
```
**Fix**: Add `canid_weight`, `nbr_weight`, `kl_weight` to `VGAEArchitecture` in `schema.py`. Reference from both sites.

### H2: Minor hardcoded values (lower priority)
- `cli.py:476` — `max_epochs < 10` smoke test threshold
- `evaluation.py:353` — `ATTENTION_SAMPLE_LIMIT = 50`
- `data_loading.py:37` — fallback `in_channels = 11`
- `temporal.py:201` — `0.8` train/val split (should use `preprocessing.train_val_split`)
- `batch_sizing.py:21` — `max(8, ...)` minimum batch size
- `sweep_pipeline.py:88` — `ResourceSpec(gpus=0, cpus=4, memory_gb=16)` ignores resources.yaml

---

## Correctness Fixes

### C1: `validate.py:46` calls `get_artifact()` instead of `artifact_exists()`
Downloads artifact just to check existence. Should use `artifact_exists()`.

### C2: `artifacts.py` — `"eval" if stage == "evaluation"` duplicated 3×
Lines 37, 43, 50 each do this mapping independently.
**Fix**: Extract `_stage_model_type(stage, model_type)` helper.

### C3: Imports inside loops
- `training.py:170`: `from graphids.core.preprocessing import get_batch_index` inside loop body
- `data_loading.py:203`: same import inside loop body
**Fix**: Move to top of function.

---

## Structural Improvements

### G1: `evaluate()` god function → per-model evaluators (305 → ~50 orchestrator + 4 evaluators)
Extract: `_evaluate_gat()`, `_evaluate_vgae()`, `_evaluate_fusion()`, `_evaluate_temporal()`, `_evaluate_test_scenarios()`. Pure code motion, no behavior change.

### G2: `_run_single_stage()` god function → lifecycle hooks (194 → ~80 + 3 helpers)
Extract: `_archive_previous()`, `_log_stage_artifacts()`, `_write_lake_manifest()`.

### O1: Shared inference scaffold
`_run_gat_inference()` and `_run_vgae_inference()` share `clone().to(device)` + `graph_label()` + `graph_attack_type()` pattern. Extract `_inference_loop()`.

### O2: Probe spatial dim
Duplicated in `temporal.py:187` and `evaluation.py:258`. Extract `probe_embedding_dim()`.

### O3: DQN checkpoint save
`fusion.py` lines 67-74 and 77-86 are near-identical. Extract `_save_dqn_checkpoint()`.

### O4: Thin wrappers in `sweep_pipeline.py`
`_checkpoint_path()` / `_metrics_path()` (lines 122-133) call `resolve()` just for a path. Delete, use config functions directly.

---

## Implementation Phases

### Phase 1 — Bugs + config + correctness (small, safe, high value)

| Item | File(s) | Change | Lines |
|------|---------|--------|-------|
| B1 | `tune_config.py:325` | `trainable_label` → `"subprocess"` | 1 |
| B2 | `job.py:30` | `"serial"` → `"cpu"` | 1 |
| H1 | `schema.py`, `modules.py:81`, `training.py:179` | VGAE loss weights to config | ~15 |
| C1 | `validate.py:46` | `get_artifact()` → `artifact_exists()` | 1 |
| C2 | `artifacts.py` | Extract `_resolve_stage_model()` helper | ~10 |
| C3 | `training.py:170`, `data_loading.py:203` | Move imports to function top | 2 |

**Est**: 30 min. **Risk**: Minimal. **All items are independent.**

### Phase 2 — `torchmetrics.MetricCollection` (replaces sklearn calls in eval)

| Item | File(s) | Change | Lines |
|------|---------|--------|-------|
| Replace `_compute_metrics()` sklearn calls | `evaluation.py` | Use `MetricCollection` with Binary* metrics | ~40 replaced, ~15 custom (detection-at-FPR, Youden's J) stay |

**Est**: 1 hour. **Risk**: Low (torchmetrics already a transitive dep). **Eliminates manual metric code, gains GPU-native computation.**

Note: This subsumes part of G1 — metrics become a reusable `MetricCollection` that G1's per-model evaluators can share.

### Phase 3 — Batched eval inference via PyG `DataLoader` + `unbatch()`

| Item | File(s) | Change | Lines |
|------|---------|--------|-------|
| Batch `_run_gat_inference()` | `evaluation.py` | Use DataLoader, unbatch per-graph | ~40 → ~25 |
| Batch `_run_vgae_inference()` | `evaluation.py` | Same pattern | ~40 → ~25 |
| Batch `cache_predictions()` | `data_loading.py` | Use DataLoader for state caching | ~30 → ~20 |

**Est**: 2 hours. **Risk**: Medium (return shapes change, attention capture stays per-sample). **10-50× eval speedup on large datasets.**

Note: This subsumes O1 (shared inference scaffold) — the batched loop IS the shared scaffold.

### Phase 4 — God function decomposition

| Item | File(s) | Change | Lines |
|------|---------|--------|-------|
| G1 | `evaluation.py` | Extract per-model evaluators | 305 → ~50 + 4 evaluators |
| G2 | `cli.py` | Extract lifecycle hooks | 194 → ~80 + 3 helpers |
| O2 | `temporal.py`, `evaluation.py` | Extract `probe_embedding_dim()` | ~10 saved |
| O3 | `fusion.py` | Extract `_save_dqn_checkpoint()` | ~10 saved |
| O4 | `sweep_pipeline.py` | Delete thin wrappers | ~12 saved |

**Est**: 2 hours. **Risk**: Low (pure code motion). **Improves testability and readability.**

Note: Phase 3 should come before Phase 4 — the batched inference changes the shape of `_run_gat_inference()` and `_run_vgae_inference()`, so decomposing `evaluate()` after batching avoids rework.

---

## Deferred / Won't Do

| Item | Reason |
|------|--------|
| submitit migration | Execution model mismatch (pickle vs CLI). Saves ~60 lines, breaks signal handling + Dagster Pipes + checkpoint resume. |
| Dagster asset checks | Valid but not urgent. Custom artifact validation works and is tested. |
| Lightning for DQN fusion | RL doesn't fit Lightning's `training_step()` paradigm. Manual loop is justified (177 lines). |
| MLflow dedup | Autolog covers training scalars only. Manual logging covers eval artifacts. Overlap is harmless. |
| Hydra/OmegaConf | Conflicts with existing Pydantic+YAML config system (`ConfigHandler`, `resolve()`). |
| Typer for CLI | Nice-to-have (~70 lines saved). CLI is stable and not growing. Low priority. |
| TorchRL `ReplayBuffer` | Saves ~40 lines but adds ~100MB dep. Custom buffer is 50 lines, correct, and has no bugs. |
| `ckatorch` for CKA | 10-line NumPy formula. Not worth a new dependency. |
| Captum for layer activations | Awkward with PyG Data objects. Custom 20-line approach is cleaner. |
| PyG `AttentionExplainer` | Aggregates across layers, losing per-layer granularity needed for paper figures. |

---

## Expected Outcome

| Phase | Lines Saved | Perf Impact | Risk |
|-------|-------------|-------------|------|
| 1 (bugs/config/correctness) | ~30 net | Bug fixes | Minimal |
| 2 (torchmetrics) | ~40 | Negligible | Low |
| 3 (batched inference) | ~30 | **10-50× eval speedup** | Medium |
| 4 (god functions) | ~30 | None (readability) | Low |
| **Total** | **~130 lines** | **Major eval speedup** | |

Post-consolidation pipeline: ~5,620 lines (from 5,750). The line count doesn't drop dramatically because the pipeline was already well-factored — the wins are in correctness (bugs, config), performance (batched eval), and maintainability (god function decomposition).
