# Preprocessing Consolidation Plan

> Status: **proposed** | Date: 2026-03-27 | Audited: 2026-03-30

Consolidate `graphids/core/preprocessing/` (1,555 lines, 8 files + 1 subdirectory) by
removing dead code, eliminating cfg duck-typing, and relocating misplaced logic.

## File inventory

| File | Lines | Verdict |
|------|-------|---------|
| `features.py` | 438 | Domain logic. Dead code: `edge_to_tensor` (0 callers), test-only paths. |
| `datamodule.py` | 413 | Lightning DataModules. SimpleNamespace cfg reconstruction × 2, misplaced staticmethod. |
| `curriculum.py` | 229 | Lightning DataModule. Delegates to parent `setup()`. Epoch tracking via callback (correct). |
| `datasets/can_bus.py` | 211 | Domain logic. One inline atomic-write duplication. |
| `_temporal.py` | 175 | **Dead** — zero production callers. |
| `utils.py` | 49 | Small utilities. Stays. |
| `__init__.py` | 37 | Re-exports + dead lazy-import for temporal symbols. |
| `datasets/__init__.py` | 3 | Pass-through. Stays. |

## 1. Delete `_temporal.py` (175 lines)

### Evidence

`TemporalDataModule`, `TemporalGrouper`, `TemporalGraphDataset`, `collate_temporal`,
`GraphSequence` have **zero production callers**. No stage YAML wires `TemporalDataModule`
to LightningCLI. The pipeline stages that used them (`train_temporal`,
`_evaluate_temporal`) were deleted during the evaluation pipeline cleanup (commit `c41511d`).

Only external reference: `tests/core/models/test_temporal.py:39` imports `collate_temporal`
directly from `_temporal` (bypasses `__init__` lazy path).

The `__init__.py` lazy `__getattr__` loader (lines 31-37), `_temporal_names` set (line 33),
and `__all__` entries (lines 21, 24, 25, 26, 27) for all 5 temporal symbols are dead.

Note: `TemporalDataModule.__init__` takes an untyped `cfg` object (not flat typed
primitives), so it would break LightningCLI / jsonargparse introspection if a stage YAML
were ever added. This reinforces that it's stale, not in-progress.

### Action

- Delete `_temporal.py` (175 lines)
- Remove temporal entries from `__init__.py` `__all__`, `_temporal_names`, and `__getattr__`
- Update or delete `test_temporal.py` collation test (move `collate_temporal` inline
  into the test if the test itself is still needed, or delete if temporal model coverage
  is handled elsewhere)
- Clean stale `.pyc` files in `__pycache__/` (temporal, dataset, engine bytecodes from prior refactors)

### Line count: **-175 lines** (+ ~10 lines from `__init__.py` cleanup)

## 2. Eliminate `SimpleNamespace` cfg reconstruction — make `load_datasets` accept kwargs

### Problem

Two DataModule `setup()` methods reconstruct a `types.SimpleNamespace` to
satisfy `load_datasets(cfg)`:

- `CANBusDataModule.setup` (`datamodule.py:217-225`) — `hp` accessed via string keys
- `FusionDataModule.setup` (`datamodule.py:382-388`) — `hp` accessed via attributes

Each builds a two-level namespace:
```python
cfg = types.SimpleNamespace(
    dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
    preprocessing=types.SimpleNamespace(
        window_size=hp.window_size, stride=hp.stride,
        train_val_split=1.0 - hp.val_fraction,
    ),
)
```

`CurriculumDataModule.setup()` (curriculum.py:160) delegates via `super().setup(stage)` to
`CANBusDataModule.setup()` — it does **not** construct its own namespace.

`load_datasets` (`datamodule.py:161`) takes a single untyped `cfg` arg and duck-types on
`cfg.dataset`, `cfg.lake_root`, `cfg.seed`, `cfg.preprocessing.window_size`,
`cfg.preprocessing.stride`, `cfg.preprocessing.train_val_split`.

Third caller: `_temporal.py:129` passes `self.cfg` directly (being deleted in section 1).

### Action

Change `load_datasets` to accept keyword args directly:

```python
def load_datasets(
    *, dataset: str, lake_root: str, seed: int,
    window_size: int, stride: int, val_fraction: float,
) -> tuple[CANBusDataset, CANBusDataset, dict[str, CANBusDataset]]:
```

Then both callers simplify to:
```python
self._train_ds, self._val_ds, self._test_datasets = load_datasets(
    dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
    window_size=hp.window_size, stride=hp.stride, val_fraction=hp.val_fraction,
)
```

No more `import types`, no nested `SimpleNamespace`.

(`CurriculumSampler.__init__` at `curriculum.py:27-55` already takes flat keyword args — no
namespace pattern to clean up there.)

### Line count: ~**-14 lines** (2 × 7-line namespace blocks → 2 × 3-line calls)

## 3. Move `cache_predictions` from `FusionDataModule` to `fusion_features.py`

### Problem

`FusionDataModule.cache_predictions` (`datamodule.py:346-372`) is a `@staticmethod` that:
- Takes models + data + device + max_samples + batch_size
- Calls `registry_extractors()` to get active extractors
- Runs model inference in batches to build state vectors
- Returns `{"states": ..., "labels": ...}`
- Has no DataModule instance state

This is model-layer logic (feature extraction) living in a data-layer class. It violates
the import hierarchy: preprocessing → models is a downward import (code-style.md says
pipeline/preprocessing imports core models lazily, but a staticmethod that runs model
inference belongs in the models package).

### Action

Move to `fusion_features.py` as a standalone function `cache_predictions(models, data,
device, max_samples, batch_size)`. After section 6 of the models plan, `extractors()` will
already live in `fusion_features.py`, making this a natural home.

`FusionDataModule.setup` calls it as:
```python
from graphids.core.models.fusion_features import cache_predictions
self.train_cache = cache_predictions(models, list(train_ds), device, hp.max_samples, ...)
```

### Line count: net 0 (moved, not deleted)

## 4. Generalize `atomic_save` for JSON writes

### Problem

`CANBusDataset._write_cache_metadata` (`can_bus.py:112-159`) reimplements the
tmpfile → fsync → rename pattern inline (lines 145-158: `tempfile.mkstemp` → `os.fdopen` →
`json.dump` + `f.flush()` + `os.fsync` → `os.rename`, with cleanup on exception).

`utils.py:34-49` has `atomic_save` using the same pattern but hardcoded to `torch.save`.

### Action

Add `atomic_write` to `utils.py` that accepts a write callable:

```python
def atomic_write(path: Path, write_fn: Callable[[Path], None]) -> None:
    """Atomic file write via tmpfile + fsync + rename. NFS-safe."""
    tmp = path.with_suffix(".tmp")
    write_fn(tmp)
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    tmp.rename(path)
```

Then `_write_cache_metadata` becomes:
```python
atomic_write(meta_path, lambda p: p.write_text(json.dumps(meta, indent=2)))
```

And `atomic_save` becomes:
```python
def atomic_save(data, path: Path) -> None:
    atomic_write(path, lambda p: torch.save(data, p))
```

### Line count: ~**-10 lines** (inline reimplementation deleted, +5 for generalized util)

## 5. Dead and test-only paths in `features.py`

### Observation

| Function | Defined at | Production callers | Test callers |
|----------|------------|-------------------|--------------|
| `clustering_coefficients` | `features.py:105` | None (called internally by `stats_to_tensor`) | `test_features.py` (6 sites) |
| `stats_to_tensor` | `features.py:124` | None (called internally by `node_features`) | None (reachable only via `node_features`) |
| `node_features` | `features.py:153` | None | `test_features.py` (6 sites) |
| `edge_to_tensor` | `features.py:191` | **None** | **None — completely dead** |

The production path (`sliding_window_graphs`, lines 201-438) uses vectorized Polars
triangle counting and stat expressions inline, never calling any of these.

`edge_to_tensor` is fully dead — zero callers in production or tests. Safe to delete now.

The `node_features` → `stats_to_tensor` → `clustering_coefficients` chain is the per-window
path used by tests for correctness validation against the vectorized production path.

**Note:** `test_features.py` imports `edge_features`, `_assemble_chunk_numpy`, and
`_numpy_to_data` — symbols that do not exist in the current `features.py`. These tests
are likely broken. Investigate before relying on test-only path as justification for
keeping code.

**Action:**
- Delete `edge_to_tensor` now (dead code, ~10 lines).
- Investigate broken test imports; fix or delete affected tests.
- Keep `node_features` chain for now pending test investigation.

## 6. Lightning DataModule convention violations

All three DataModules deviate from the `LightningDataModule` contract
(Lightning docs: `data/datamodule.rst`). These aren't style nits — they
block DDP and skip unnecessary work.

### 6a. No `prepare_data()` / `setup()` separation — all three DataModules

Lightning convention:
- `prepare_data()` — single process, one-time work (download, cache). **No `self.x = y`.**
- `setup(stage)` — every GPU, per-stage splits. Safe to assign `self.x`.

Current code puts heavy one-time work in `setup()`:

| DataModule | Heavy work in `setup()` | Should be in `prepare_data()` |
|---|---|---|
| `CANBusDataModule` | `load_datasets()` → `CANBusDataset.process()` (NFS-locked CSV scan + graph building) | Yes — idempotent one-time cache build |
| `FusionDataModule` | Loads VGAE + GAT models, runs inference (`cache_predictions`) | Yes — one-time state vector caching |
| `CurriculumDataModule` | Delegates to `CANBusDataModule.setup()`, then loads VGAE and scores difficulty | Yes — one-time scoring |

Single-GPU (current) works fine. DDP would run redundantly per-GPU or race on NFS.

**Action:** Split each DataModule into:

```python
def prepare_data(self):
    # One-time: build cache if not exists, score difficulty, etc.
    # No self.x = y assignments

def setup(self, stage):
    # Per-GPU: load already-cached data into self._train_ds, etc.
```

For `CANBusDataModule`, `CANBusDataset.process()` already has NFS locking and a `.complete`
marker — it's naturally idempotent. The split is: `prepare_data()` triggers
`CANBusDataset(root, ...)` to ensure cache exists, `setup()` loads the cached `.pt` files.

For `FusionDataModule` and `CurriculumDataModule`, the model inference results can be cached
to disk in `prepare_data()` and loaded in `setup()`.

### 6b. `setup()` ignores `stage` — all three DataModules

Lightning passes `stage="fit"`, `"test"`, or `"predict"`. Current code loads everything
unconditionally.

**Action:** Guard by stage:

```python
def setup(self, stage):
    if stage in ("fit", None):
        self._train_ds = ...
        self._val_ds = ...
    if stage in ("test", None):
        self._test_datasets = ...
```

Avoids loading train/val data when only testing, and vice versa.

### 6c. No `predict_dataloader()` — all three DataModules

Models define `predict_step()` but `trainer.predict(datamodule=dm)` would fail — no
`predict_dataloader()` method. `CANBusDataModule` has `test_dataloader()` (line 282) but
that returns a list of loaders per attack type, which isn't what predict needs.

`FusionDataModule` has `train_dataloader` (line 407) and `val_dataloader` (line 411).
`CurriculumDataModule` has `train_dataloader` (line 211) and `val_dataloader` (line 226).

**Action:** Add `predict_dataloader()` to `CANBusDataModule` — returns the test loader(s)
or a dedicated predict split. Low priority but completes the API.

### ~~6d. `_current_epoch` tracking~~ — RESOLVED

~~`CurriculumDataModule` tracks `_current_epoch` manually.~~

**Already fixed.** Epoch advancement uses `CurriculumEpochCallback.on_train_epoch_start()`
(`curriculum.py:101-104`) which calls `dm._batch_sampler.set_epoch(trainer.current_epoch)`.
No manual counter exists. `state_dict`/`load_state_dict` are unnecessary since
`trainer.current_epoch` is already checkpointed by Lightning.

### ~~6e. Side effects in `train_dataloader()`~~ — PARTIALLY RESOLVED

~~Manual `_current_epoch` counter incremented in `train_dataloader()`.~~

**The manual epoch counter is gone.** However, `train_dataloader()` (`curriculum.py:211-224`)
still has a first-call side effect: when `_batch_sampler.max_num_nodes is None`, it calls
`vram_node_budget()` and writes to `self._batch_sampler.max_num_nodes`,
`self._batch_sampler.mean_nodes`, and rebuilds `self._batch_sampler._inner`. This is
acceptable for now (single-GPU, first-call only).

### ~~6f. `compute_node_budget` in models layer~~ — RESOLVED

~~`compute_node_budget` + `NodeBudgetInfo` in `_training.py`, imported by DataModules.~~

**Already gone.** Neither `compute_node_budget` nor `NodeBudgetInfo` exist anywhere in the
codebase. `vgae.py` does not import either symbol. No action needed.

### Line count estimate (convention fixes)

| Change | Added | Deleted |
|--------|-------|---------|
| `prepare_data()` / `setup()` split (3 DMs) | +30 | -10 |
| `stage` guards in `setup()` | +12 | 0 |
| `predict_dataloader()` on `CANBusDataModule` | +3 | 0 |
| **Section net** | **+45** | **-10** |
| **Section delta** | | **+35 lines** (correctness cost) |

## Execution order

1. Delete `_temporal.py` + clean `__init__.py` (standalone, no deps)
2. Delete `edge_to_tensor` from `features.py` (standalone, zero callers)
3. Investigate broken test imports in `test_features.py` (`edge_features`, `_assemble_chunk_numpy`, `_numpy_to_data`)
4. Refactor `load_datasets` to keyword args, update 2 callers
5. Move `cache_predictions` to `fusion_features.py` (depends on models plan section 6)
6. Generalize `atomic_write` in `utils.py`, simplify `_write_cache_metadata`
7. `prepare_data()` / `setup(stage)` split on all 3 DataModules
8. Add `predict_dataloader()` to `CANBusDataModule`
9. Verify: import checks + `--collect-only`

## Combined line count

| Section | Delta |
|---------|-------|
| Delete `_temporal.py` + `__init__.py` cleanup (section 1) | -185 |
| Delete `edge_to_tensor` (section 5) | -10 |
| `load_datasets` kwargs (section 2) | -14 |
| `cache_predictions` relocation (section 3) | 0 |
| `atomic_write` generalization (section 4) | -5 |
| Convention fixes — 6a/6b/6c only (section 6) | +35 |
| **Total** | **-179 lines** |

## Risks

- **`load_datasets` callers outside preprocessing:** Verified — only `CANBusDataModule` +
  `FusionDataModule` + `_temporal.py` (being deleted) call it. `CurriculumDataModule`
  delegates via `super()`. No external callers.
- **`collate_temporal` in test:** One test (`test_temporal.py:39`) imports it directly from
  `_temporal` (bypasses `__init__` lazy path). Either inline in the test or delete
  the test if temporal model tests are covered by `test_temporal.py`'s other test methods
  that exercise `TemporalLightningModule` directly.
- **`cache_predictions` move timing:** Depends on models plan section 6 (registry dissolution)
  so that `extractors()` is already in `fusion_features.py`. Execute after that section.
- **Broken test imports:** `test_features.py` imports symbols that don't exist in `features.py`
  (`edge_features`, `_assemble_chunk_numpy`, `_numpy_to_data`). These tests may already be
  failing silently — investigate before deleting the test-only code paths they exercise.
