# Pipeline Cleanup Plan

Audit of `graphids/pipeline/` — reinvented wheels, internal duplication, and bloat.

## Reinvented Wheels (package API exists)

| What | Where | Lines | Use Instead |
|------|-------|------:|-------------|
| CKA implementation | `cka.py:12-26` | 15 | `torch-cka` 0.21 (already installed) |
| Balanced accuracy post-hoc | `eval_inference.py:83` | 1 | `torchmetrics.BinaryBalancedAccuracy` in each module's MetricCollection |
| Node-budget packing loop | `modules.py:421-443` | 23 | PyG `DynamicBatchSampler` (already imported in data_loading.py) |
| `_lerp()` | `modules.py:78-80` | 3 | `math.lerp` (stdlib 3.9+) |
| Focal loss | `modules.py:64-70` | 7 | `torchvision.ops.sigmoid_focal_loss` (verify multiclass form) |

## Internal Duplication

| What | Where | Wasted | Fix |
|------|-------|-------:|-----|
| 4 identical fusion train fns | `fusion.py:223-303` | ~60 | Extract `_train_fusion_method(module, save_fn, ...)` |
| `eval_vgae` / `eval_dgi` twins | `evaluation.py:155-224` | ~30 | Extract `_eval_unsupervised(module, capture_fn=...)` |
| `make_projection` duplicate | `trainer_factory.py:119-134` | 16 | Delete standalone, call from `prepare_kd` |
| Identity hash re-extraction | `stages/__init__.py:92-93` | 2 | Read `cfg.identity_hash` via existing OmegaConf resolver |

## eval_inference.py Issues

| Issue | Where | Fix |
|-------|-------|-----|
| `test_model` returns `{"core": ..., "additional": {}}` — `additional` always empty, every caller indexes `["core"]` | `:66-84` | Return flat metrics dict |
| `find_vgae_threshold` reaches into module privates (`_test_errors`, `_test_labels`) | `:91-132` | Move threshold search to module method, or expose accumulation as interface |
| `run_fusion_inference` duck-types agent via `hasattr(agent, "q_network")` | `:237-257` | Add `agent.get_q_values(states)` to common interface |

## evaluation.py Issues

| Issue | Where | Fix |
|-------|-------|-----|
| `eval_vgae`/`eval_dgi` near-identical (70 lines duplicated) | `:155-224` | See duplication table above |
| `eval_fusion` 4-way if/elif reconstructing agents from checkpoints | `:227-300` | Add `from_checkpoint()` classmethod to each model class |
| `eval_temporal` bare `except Exception` swallows all errors | `:303-354` | Catch specific exceptions or at least `log.exception()` |

## data_loading.py — DELETE ENTIRE FILE

Junk drawer of unrelated helpers grouped under "data loading" by prior sessions. No function justifies the file's existence.

| Function | Lines | Disposition |
|----------|------:|-------------|
| `NodeBudgetInfo` | 4 | Used in one place — inline as tuple or move to `modules.py` |
| `cleanup()` | 5 | 3-line `gc.collect() + empty_cache()` wrapper — inline at call sites |
| `compute_node_budget()` | 27 | Reads one JSON, multiplies two numbers — inline or move to `modules.py` (sole budget consumer) |
| `make_dataloader()` | 42 | Thin wrapper over `PyGDataLoader` constructor — inline the 5-line PyG call at each call site, put spawn defaults in a constant |
| `cache_predictions()` | 27 | Fusion-specific, not data loading — move to `fusion.py`. Also defeats its own batching: `batch.to_data_list()` in inner loop unbatches immediately after DataLoader batches |

**Callers to update:** `training.py`, `modules.py`, `evaluation.py`, `fusion.py`, `temporal.py`

## callbacks.py — DELETE ENTIRE FILE

Misnamed file: 3 of 4 functions are plain serialization helpers, not callbacks. Single consumer (`evaluation.py` lines 96-102). No reuse.

| Function | Lines | Disposition |
|----------|------:|-------------|
| `save_embeddings()` | 21 | Unpacks dataclass → `np.savez_compressed`. Move to `evaluation.py` as private or inline |
| `save_attention()` | 16 | Flattens attention dicts → `np.savez_compressed`. Move to `evaluation.py` as private or inline |
| `save_dqn_policy()` | 16 | Partitions alphas by label → `json.dumps`. Move to `evaluation.py` as private or inline |
| `RunMetadataCallback` | 19 | Fires `git rev-parse HEAD` on `on_fit_end` → writes JSON. Move to 3 lines in `run_stage()` which already sets up the run dir |

**Callers to update:** `evaluation.py` (only consumer), `trainer_factory.py` (imports `RunMetadataCallback`)

## Pointless IO — Hydra/Lightning Already Handles This

| Pointless IO | Where | Lines | What Handles It |
|-------------|-------|------:|-----------------|
| Manual mkdir + chdir into run dir | `__init__.py:45-47` | 3 | `hydra.job.chdir=true` (default since Hydra 1.2+) |
| Manual config.yaml save | `__init__.py:55` | 1 | Hydra auto-saves `.hydra/config.yaml`, `.hydra/overrides.yaml`, `.hydra/hydra.yaml` |
| DuckDB catalog — hand-built schema, manual INSERT, string-split identity hash | `__init__.py:61-113` | 53 | mlflow 3.10.1 (already installed) via Lightning `MLFlowLogger`, or CSVLogger. Catalog is self-described as "disposable — rebuildable from filesystem" |
| 4x `_restore_best_weights` → `torch.save("best_model.pt")` | `fusion.py:220,263,284,301` | ~30 | `ModelCheckpoint` — if agents implement proper `state_dict`/`load_state_dict` |
| `torch.save(temporal_model.state_dict(), "best_model.pt")` | `temporal.py:289` | 1 | `ModelCheckpoint` |
| `RunMetadataCallback` — subprocess git SHA → JSON | `callbacks.py:98-106` | 19 | Config field resolved at startup, or `mlflow.log_param("git_sha", ...)` |
| `metrics.json` manual write — re-serializes metrics already logged via `self.log()` | `evaluation.py:109` | 1 | Lightning CSVLogger / mlflow. Current path: module logs → Lightning collects → `trainer.test()` returns → `evaluate()` re-wraps → writes JSON (3 unnecessary hops) |

**Total:** ~108 lines of IO code that duplicates framework behavior.

### Decision needed: mlflow vs CSVLogger

mlflow 3.10.1 is installed but undeclared in pyproject.toml and unused. Two paths:
1. **Adopt mlflow:** Add `MLFlowLogger` to trainers, delete DuckDB catalog, delete metrics.json write. Declare in pyproject.toml.
2. **Stay with CSVLogger:** Already works via Lightning. Delete DuckDB catalog. Keep metrics.json only if paper_sync.py needs it (check).

## Undeclared Dependencies

| Package | Version | Status |
|---------|---------|--------|
| mlflow | 3.10.1 | Installed, not in pyproject.toml |
| dagster | 1.12.19 | Installed, not in pyproject.toml — overlaps orchestration/manifest.py |
| fastapi + uvicorn | 0.131.0 / 0.41.0 | Installed, unused by pipeline |
| torchmetrics | 1.8.2 | Directly imported but not declared (transitive via Lightning) |

## Phase 2: Pipeline↔Model Boundary Fixes

Audit found 5 cases where pipeline reimplements model logic. Root cause: DQN agent was built first with primitives, bandit was built properly (self-contained), and nobody went back to fix DQN to match.

### True Duplications

| # | What | Copies | Fix |
|---|------|-------:|-----|
| 2 | Fused score formula (select→derive→blend→threshold) | 3 | `dqn.validate_batch` is canonical. Delete copies in `fusion.py:test_step:85-91` and `eval_inference.py:run_fusion_inference:241-244` — call `agent.validate_batch()` or new `agent.predict()` |
| 3 | DQN checkpoint save format | 2 | `dqn.py:load_checkpoint` knows the format but DQN has no `state_dict()`. Add it (bandit already has one at `bandit.py:349`). Delete manual dict in `fusion.py:224-228` |
| 6 | VGAE component loss decomposition | 2 | `modules.py:VGAEModule._task_loss:177-194` is canonical. `eval_inference.py:capture_vgae_artifacts:196-221` is a copy. **DIVERGED: _task_loss uses scatter reduce="max", copy uses reduce="mean"** — silent bug where eval artifacts don't match training metrics |
| 8 | Fusion baseline checkpoint load (4-way if/elif) | 1 | `evaluation.py:226-254` manually reconstructs each method. Add `from_checkpoint(ckpt, cfg)` to `MLPFusionModule` and `WeightedAvgModule` |

### Missing DQN Methods (bandit has them)

| Method | Bandit | DQN | Action |
|--------|--------|-----|--------|
| `train_episode()` | `bandit.py:260` | Missing | Add — move `_dqn_train_step` logic from pipeline into model |
| `state_dict()` | `bandit.py:349` | Missing | Add — mirror bandit's pattern |
| `predict(states)` | via `validate_batch:313` | via `validate_batch:286` | **Exists but pipeline doesn't call it** — 3 copies of the formula instead |

### Execution Order

1. Add `DQNAgent.state_dict()` + `DQNAgent.train_episode()` — model becomes self-contained like bandit
2. Add `predict()` to both agents — delete 3 copies of fused score formula
3. Add `from_checkpoint()` to fusion baselines — delete 4-way if/elif in evaluation.py
4. Fix scatter reduce divergence in `capture_vgae_artifacts` — `mean` vs `max` silent bug
5. Unify trainer factories — fusion trainer config into Hydra config, `make_trainer` handles all stages

### Trainer Factory Consolidation

`trainer_factory.py:make_trainer()` and `fusion.py:_make_fusion_trainer()` both build `pl.Trainer` with callbacks/logger/accelerator. Fusion callbacks (ModelCheckpoint, EarlyStopping) should be declared in Hydra config like main stages, then `make_trainer` handles everything and `_make_fusion_trainer` is deleted.
