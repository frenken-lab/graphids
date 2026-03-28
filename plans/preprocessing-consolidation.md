# Preprocessing Consolidation Plan

> Status: **proposed** | Date: 2026-03-27

Consolidate `graphids/core/preprocessing/` (1,406 lines, 8 files + 1 subdirectory) by
removing dead code, eliminating cfg duck-typing, and relocating misplaced logic.

## File inventory

| File | Lines | Verdict |
|------|-------|---------|
| `features.py` | 438 | Domain logic. Partial dead code (test-only paths). |
| `datasets/can_bus.py` | 211 | Domain logic. One inline atomic-write duplication. |
| `datamodule.py` | 309 | Lightning DataModules. SimpleNamespace cfg reconstruction × 2, misplaced staticmethod. |
| `curriculum.py` | 184 | Lightning DataModule. SimpleNamespace cfg reconstruction × 1. |
| `_temporal.py` | 175 | **Dead** — zero production callers. |
| `utils.py` | 49 | Small utilities. Stays. |
| `__init__.py` | 37 | Re-exports + dead lazy-import for temporal symbols. |
| `datasets/__init__.py` | 3 | Pass-through. Stays. |

## 1. Delete `_temporal.py` (175 lines)

### Evidence

`TemporalDataModule`, `TemporalGrouper`, `TemporalGraphDataset`, `collate_temporal` have
**zero production callers**. The pipeline stages that used them (`train_temporal`,
`_evaluate_temporal`) were deleted during the evaluation pipeline cleanup (commit `c41511d`).

Only caller is `tests/core/models/test_temporal.py:39` which imports `collate_temporal`
for a unit test of the temporal model's collation — this test should either be deleted
or import the collation inline.

The `__init__.py` lazy `__getattr__` loader (lines 30-36) and `__all__` entries
(lines 21, 24, 26-27) for temporal symbols are also dead.

### Action

- Delete `_temporal.py` (175 lines)
- Remove temporal entries from `__init__.py` `__all__` and `__getattr__`
- Update or delete `test_temporal.py` collation test (move `collate_temporal` inline
  into the test if the test itself is still needed, or delete if temporal model coverage
  is handled elsewhere)

### Line count: **-175 lines** (+ ~10 lines from `__init__.py` cleanup)

## 2. Eliminate `SimpleNamespace` cfg reconstruction — make `load_datasets` accept kwargs

### Problem

Three DataModule `setup()` methods reconstruct an identical `types.SimpleNamespace` to
satisfy `load_datasets(cfg)`:

- `CANBusDataModule.setup` (`datamodule.py:116-124`)
- `FusionDataModule.setup` (`datamodule.py:277-284`)
- `CurriculumDataModule.setup` (`curriculum.py:123-130`)

Each builds:
```python
cfg = types.SimpleNamespace(
    dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
    preprocessing=types.SimpleNamespace(
        window_size=hp.window_size, stride=hp.stride,
        train_val_split=1.0 - hp.val_fraction,
    ),
)
```

This exists because `load_datasets` duck-types on `cfg.dataset`, `cfg.lake_root`,
`cfg.seed`, `cfg.preprocessing.window_size`, etc.

### Action

Change `load_datasets` to accept keyword args directly:

```python
def load_datasets(
    *, dataset: str, lake_root: str, seed: int,
    window_size: int, stride: int, val_fraction: float,
) -> tuple[CANBusDataset, CANBusDataset, dict[str, CANBusDataset]]:
```

Then all three callers simplify to:
```python
self._train_ds, self._val_ds, self._test_datasets = load_datasets(
    dataset=hp.dataset, lake_root=hp.lake_root, seed=hp.seed,
    window_size=hp.window_size, stride=hp.stride, val_fraction=hp.val_fraction,
)
```

No more `import types`, no nested `SimpleNamespace`.

Also removes `CurriculumSampler.__init__`'s separate `sampler_cfg` namespace
(`curriculum.py:152`) if it uses the same pattern.

### Line count: ~**-20 lines** (3 × 7-line namespace blocks → 3 × 3-line calls)

## 3. Move `cache_predictions` from `FusionDataModule` to `fusion_features.py`

### Problem

`FusionDataModule.cache_predictions` (`datamodule.py:243-268`) is a `@staticmethod` that:
- Takes models + data + device
- Imports `registry_extractors()` from `graphids.core.models.registry`
- Runs model inference to build state vectors
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

`CANBusDataset._write_cache_metadata` (`can_bus.py:112-158`) reimplements the
tmpfile → fsync → rename pattern inline for JSON output. `utils.py` has `atomic_save`
but it hardcodes `torch.save`.

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

## 5. Dead test-only paths in `features.py`

### Observation (not an action item yet)

`clustering_coefficients`, `stats_to_tensor`, `node_features`, `edge_to_tensor` (~65 lines)
are only called from tests (`test_features.py`), not from the production path
(`sliding_window_graphs`). The production path uses vectorized Polars triangle counting
inline.

These are the per-window path (process one window at a time) vs the batch production path
(process all windows vectorized). The per-window path is useful for tests and debugging.

**Decision:** Keep for now. Flag for future cleanup if test coverage migrates to
testing `sliding_window_graphs` directly. Not blocking.

## 6. Lightning DataModule convention violations

All three DataModules deviate from the `LightningDataModule` contract
(Lightning docs: `data/datamodule.rst`). These aren't style nits — they
block DDP, break checkpoint resume, and cause side-effect bugs.

### 6a. No `prepare_data()` / `setup()` separation — all three DataModules

Lightning convention:
- `prepare_data()` — single process, one-time work (download, cache). **No `self.x = y`.**
- `setup(stage)` — every GPU, per-stage splits. Safe to assign `self.x`.

Current code puts heavy one-time work in `setup()`:

| DataModule | Heavy work in `setup()` | Should be in `prepare_data()` |
|---|---|---|
| `CANBusDataModule` | `load_datasets()` → `CANBusDataset.process()` (NFS-locked CSV scan + graph building) | Yes — idempotent one-time cache build |
| `FusionDataModule` | Loads VGAE + GAT models, runs inference (`cache_predictions`) | Yes — one-time state vector caching |
| `CurriculumDataModule` | Loads VGAE, scores difficulty on all normal graphs | Yes — one-time scoring |

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
`predict_dataloader()` method. `CANBusDataModule` has `test_dataloader()` but that
returns a list of loaders per attack type, which isn't what predict needs.

**Action:** Add `predict_dataloader()` to `CANBusDataModule` — returns the test loader(s)
or a dedicated predict split. Low priority but completes the API.

### 6d. No `state_dict()` / `load_state_dict()` on `CurriculumDataModule`

`CurriculumDataModule` tracks `_current_epoch` (`curriculum.py:111`) which controls
curriculum progression (what difficulty percentile of data is active). On checkpoint
resume, this state is **lost** — training restarts curriculum from epoch 0, undoing
the progression schedule.

**Action:**

```python
def state_dict(self):
    return {"current_epoch": self._current_epoch}

def load_state_dict(self, state_dict):
    self._current_epoch = state_dict["current_epoch"]
```

Evidence: Lightning docs `extensions/datamodules_state.rst` — `state_dict`/`load_state_dict`
are automatically called by the checkpointing system.

### 6e. Side effects in `train_dataloader()` — `CurriculumDataModule`

```python
def train_dataloader(self):
    self._batch_sampler.set_epoch(self._current_epoch)  # side effect
    self._current_epoch += 1                             # mutation
    return self._train_loader
```

Lightning can call `train_dataloader()` multiple times (sanity validation check calls it,
then the real training loop calls it again). The manual counter increments incorrectly.

**Action:** Replace `self._current_epoch` with `self.trainer.current_epoch`:

```python
def train_dataloader(self):
    self._batch_sampler.set_epoch(self.trainer.current_epoch)
    return self._train_loader
```

Eliminates the manual counter entirely. The `state_dict` (6d) is then also unnecessary
since `trainer.current_epoch` is already checkpointed by Lightning.

### 6f. `compute_node_budget` lives in models layer

`datamodule.py:187` and `curriculum.py:10,149` import
`from graphids.core.models._training import compute_node_budget`. This function reads
dataset metadata and computes batch sizing — a preprocessing concern, not a model concern.

**Action:** Move `compute_node_budget` + `NodeBudgetInfo` from `_training.py` to
`datamodule.py` (or a new `_batching.py` if `datamodule.py` gets too large). The only
non-preprocessing caller is `vgae.py` which imports it — that import reverses to
`from graphids.core.preprocessing.datamodule import compute_node_budget` (allowed:
models can import preprocessing utilities).

Wait — code-style.md says core/ imports config.constants only, never pipeline. But
preprocessing is also in core/. Let me check: `vgae.py` only *imports* the symbol at
module level (`from ._training import compute_node_budget`), it doesn't *call* it.
Grepping confirms `vgae.py` imports but never calls `compute_node_budget`. So the only
actual callers are in preprocessing — the `vgae.py` import is dead.

**Action (revised):** Move to preprocessing, delete the dead import from `vgae.py`.

### Line count estimate (convention fixes)

| Change | Added | Deleted |
|--------|-------|---------|
| `prepare_data()` / `setup()` split (3 DMs) | +30 | -10 |
| `stage` guards in `setup()` | +12 | 0 |
| `predict_dataloader()` on `CANBusDataModule` | +3 | 0 |
| `CurriculumDataModule` use `trainer.current_epoch` | +1 | -5 |
| Move `compute_node_budget` to preprocessing | +2 | -2 |
| **Section net** | **+48** | **-17** |
| **Section delta** | | **+31 lines** (correctness cost) |

## Execution order

1. Delete `_temporal.py` + clean `__init__.py` (standalone)
2. Refactor `load_datasets` to keyword args, update 3 callers
3. Move `cache_predictions` to `fusion_features.py` (depends on models plan section 6)
4. Generalize `atomic_write` in `utils.py`, simplify `_write_cache_metadata`
5. Move `compute_node_budget` from `_training.py` to preprocessing, delete dead `vgae.py` import
6. `prepare_data()` / `setup(stage)` split on all 3 DataModules
7. `CurriculumDataModule`: replace manual epoch counter with `trainer.current_epoch`
8. Add `predict_dataloader()` to `CANBusDataModule`
9. Verify: import checks + `--collect-only`

## Combined line count

| Section | Delta |
|---------|-------|
| Delete `_temporal.py` + `__init__.py` cleanup (section 1) | -185 |
| `load_datasets` kwargs (section 2) | -20 |
| `cache_predictions` relocation (section 3) | 0 |
| `atomic_write` generalization (section 4) | -5 |
| Convention fixes (sections 5-8) | +31 |
| **Total** | **-179 lines** |

## Risks

- **`load_datasets` callers outside preprocessing:** Verified — only the 3 DataModules
  + `_temporal.py` (being deleted) call it. No external callers.
- **`collate_temporal` in test:** One test imports it. Either inline in the test or delete
  the test if temporal model tests are covered by `test_temporal.py`'s other test methods
  that exercise `TemporalLightningModule` directly.
- **`cache_predictions` move timing:** Depends on models plan section 6 (registry dissolution)
  so that `extractors()` is already in `fusion_features.py`. Execute after that section.
