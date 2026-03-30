# Models Consolidation Plan

> Status: **proposed** | Created: 2026-03-27 | Audited: 2026-03-30

Consolidate `graphids/core/models/` (3,287 lines, 13 files) by adopting Lightning/torchmetrics
built-ins instead of hand-rolled shared logic. Three concerns: optimizers, setup, threshold.
Plus: registry dissolution, _training.py cleanup, DQN/Bandit Lightning conversion, temporal fix.

## 1. Optimizers — delete `configure_optimizers` from GAT/DGI, refactor VGAE

LightningCLI creates optimizers automatically from `--optimizer` / `--lr_scheduler` CLI args
when the module doesn't define `configure_optimizers`.

**Action:** Add defaults in `GraphIDSCLI.add_arguments_to_parser` (`cli.py`):

```python
parser.add_optimizer_args(torch.optim.Adam)
parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
```

Then delete `configure_optimizers` from:
- `gat.py:311-314` — standard `self.parameters()`
- `dgi.py:284-287` — standard `self.parameters()`

**VGAE is not identical** — `vgae.py:530-536` conditionally adds projection parameters:
```python
params = list(self.model.parameters())
if self.projection is not None:
    params += list(self.projection.parameters())
```
Two options:
- (a) Delete it — CLI uses `self.parameters()` which includes projection. Accept tiny overhead
  from frozen teacher params getting Adam state.
- (b) Keep a minimal override that filters `requires_grad` params.

Recommend (a) for simplicity unless profiling shows meaningful memory cost on V100 16GB.

**Exceptions that keep custom `configure_optimizers`:**
- `TemporalLightningModule` (`temporal.py:320-340`) — two param groups with different LRs
- `RLFusionModule` (`fusion_baselines.py:277-278`) — manual optimization, proxies agent's optimizer

YAML configs add `optimizer:` / `lr_scheduler:` sections instead of hardcoding in Python.

## 2. Setup — shared base class for lazy model construction

VGAE (`vgae.py:388`), GAT (`gat.py:244`), DGI (`dgi.py:192`) all have identical `setup()`:

```python
def setup(self, stage=None):
    if self.model is None:
        dm = self.trainer.datamodule
        self.hparams.num_ids = dm.num_ids
        self.hparams.in_channels = dm.in_channels
        self.hparams.num_classes = dm.num_classes
        self._build()
```

**Action:** Create `GraphModuleBase` in `_training.py`:

```python
class GraphModuleBase(pl.LightningModule):
    """Shared base for VGAE, GAT, DGI — setup, OOM guard, metrics, threshold."""

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            self.hparams.num_ids = dm.num_ids
            self.hparams.in_channels = dm.in_channels
            self.hparams.num_classes = dm.num_classes
            self._build()

    def _build(self):
        raise NotImplementedError

    # OOM guard (folds OOMSkipMixin)
    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None
```

Then `VGAEModule(GraphModuleBase)`, `GATModule(GraphModuleBase)`, `DGIModule(GraphModuleBase)`.
Delete `OOMSkipMixin` class (current 3 subclasses absorbed by `GraphModuleBase`).

## 3. Threshold — `BinaryROC` replaces manual accumulation

VGAE (`vgae.py:484-517`) and DGI (`dgi.py:238-271`) both maintain manual `_test_scores` /
`_test_labels` lists, concatenate at epoch end, run Youden-J threshold selection.
Structurally identical (differ only in local variable name: `errors` vs `scores`).

**Action:** Add `BinaryROC` metric + `_find_threshold()` to `GraphModuleBase`:

```python
from torchmetrics.classification import BinaryROC

def _init_threshold_test(self):
    self.test_threshold: float | None = None
    self.roc_metric = BinaryROC()

def _find_threshold(self):
    fpr, tpr, thresholds = self.roc_metric.compute()
    if thresholds.numel() < 2:
        return None
    j = tpr - fpr
    best = torch.argmax(j)
    return float(thresholds[best]) if best < len(thresholds) else None
```

VGAE/DGI `test_step` simplifies to:
```python
def test_step(self, batch, _idx):
    scores = self._per_graph_errors(batch)  # or _per_graph_scores
    self.roc_metric.update(scores, batch.y)
    self.test_metrics.update((scores >= self.test_threshold).long(), batch.y)
```

GAT and Temporal don't use threshold (supervised). They inherit `GraphModuleBase` but
don't call `_init_threshold_test()`.

## 4. Dead code deletion

| What | Where | Why dead |
|------|-------|----------|
| `MLPFusionModule.fuse()` | `fusion_baselines.py:89-94` | References undefined `np` (numpy not imported), zero callers |
| `WeightedAvgModule.fuse()` | `fusion_baselines.py:161-168` | Same — `np` not imported, zero callers |

## 5. Registry dissolution

`registry.py` (85 lines) couples two unrelated concerns in `_MODELS`:
1. Module class resolution (`get_module_cls`) — 1 caller (`safe_load_checkpoint` in `_training.py:99`)
2. Fusion state layout (`fusion_state_dim`, `feature_layout`, `extractors`) — domain logic

**Actions:**

**Replace `get_module_cls()`** — inline a simple dict in `_training.py` next to `safe_load_checkpoint`:

```python
_MODULE_PATHS: dict[str, str] = {
    "vgae": "graphids.core.models.vgae.VGAEModule",
    "gat": "graphids.core.models.gat.GATModule",
    "dgi": "graphids.core.models.dgi.DGIModule",
    "fusion": "graphids.core.models.fusion_baselines.RLFusionModule",
}
```

**Move fusion layout to `fusion_features.py`** — `FeatureLayout`, `fusion_state_dim()`,
`feature_layout()`, `extractors()`. This file already defines the extractor Protocol
and implementations. Registration order coupling becomes explicit:

```python
_EXTRACTORS: list[tuple[str, FusionFeatureExtractor]] = [
    ("vgae", VGAEFusionExtractor()),
    ("gat", GATFusionExtractor()),
]
# Order is load-bearing — matches 15-D state layout that trained DQN checkpoints expect.
```

**Delete `registry.py`** + lazy loaders (`_vgae_module`, `_gat_module`, `_dgi_module`, `_fusion_module`).

**Update `__init__.py`** re-exports to point at `fusion_features.py`.

**Callers to update:**
- `fusion_state_dim()`: 4 production sites (`bandit.py`, `dqn.py`, `fusion_baselines.py`) + 4 test files
- `feature_layout()`: 2 sites (`fusion_baselines.py:119`, `fusion_reward.py:61`)
- `extractors()`: 1 site (`datamodule.py:366`)

## 6. `_training.py` cleanup

`_training.py` (172 lines, 7 functions) after Section 2 absorbs `OOMSkipMixin` into `GraphModuleBase`:

| Function | Lines | Callers | Verdict |
|----------|-------|---------|---------|
| `teacher_on_device()` | 14 | 2 (`gat.py:272`, `vgae.py:436`) | **Keep** |
| `safe_load_checkpoint()` | 20 | internal (via `load_inner_model`) | **Keep** |
| `load_inner_model()` | 8 | 5 sites (`_training.py`, `temporal.py`, `curriculum.py`, `datamodule.py` ×2) | **Keep** |
| `prepare_kd()` | 39 | 2 (`gat.py:259`, `vgae.py:403`) | **Keep** |
| `soft_label_kd_loss()` | 7 | 1 (`gat.py:275`) | **Inline** at call site |
| `focal_loss()` | 5 | 1 (`gat.py:238`) | **Move** to `gat.py` as `_focal_loss` |
| `binary_test_metrics()` | 12 | 7 sites (3 core + 3 fusion + temporal) | **Inline** in `GraphModuleBase.__init__` for core; keep as factory for fusion modules |
| `OOMSkipMixin` | 11 | 3 (→ absorbed by `GraphModuleBase`) | **Delete** class |
| `KDAuxiliary` | TypedDict | config schema | **Keep** |

Note: `binary_test_metrics` has 7 callers, not 6. Fusion modules (`MLPFusionModule`,
`WeightedAvgModule`, `RLFusionModule`) also call it, so keep the factory function
for fusion and inline in `GraphModuleBase` for core modules — or keep the factory for all.

## 7. `temporal.py` — use `load_inner_model` for GAT checkpoint loading

`temporal.py:243-250` manually unwraps a Lightning checkpoint:

```python
checkpoint = torch.load(gat_ckpt_path, map_location="cpu", weights_only=True)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    raw = checkpoint["state_dict"]
    checkpoint = {k.replace("model.", ""): v for k, v in raw.items() if k.startswith("model.")}
gat.load_state_dict(checkpoint)
```

**Action:** Replace with `load_inner_model("gat", gat_ckpt_path, "cpu")`. Deletes 8 lines.

`TemporalLightningModule` does NOT subclass `GraphModuleBase` — eager build in `__init__`,
no OOM guard, no threshold. Its custom `configure_optimizers` (two param groups) stays.

## 8. DQN/Bandit — eliminate wrapper, adopt Lightning primitives

`EnhancedDQNFusionAgent` (`dqn.py`) and `NeuralLinUCBAgent` (`bandit.py`) are plain Python
classes that re-implement Lightning concerns: optimizer (`dqn.py:156`), scheduler (`dqn.py:157`),
grad clipping (`dqn.py:260`, `bandit.py:223`), checkpointing (`dqn.py:349`, `bandit.py:345`).
`RLFusionModule` (`fusion_baselines.py:171-279`) wraps them with pure indirection.

**Action:** Make DQN/Bandit `LightningModule`s, delete `RLFusionModule`.

**`DQNFusionModule(FusionModuleBase)`:**
- `automatic_optimization = False` (multiple gradient steps per `training_step`)
- `configure_optimizers` returns `AdamW` + `ReduceLROnPlateau`
- Target network update via `on_train_batch_end`
- Delete `state_dict()`, `load_checkpoint()`, manual optimizer creation

**`BanditFusionModule(FusionModuleBase)`:**
- `A_inv`, `b`, `theta` → `register_buffer()` → auto-checkpointed + auto device transfer
- Delete `state_dict()`, `load_checkpoint()`, manual `.to(device)` calls

**`FusionModuleBase(pl.LightningModule)`:**
- Shared `test_step`, `on_test_epoch_start/end`, `validate_batch`, `predict`

**Observation:** gamma=0 is hardcoded (`dqn.py:119`), making `targets = rewards` unconditional
(`dqn.py:251-254`). Target network has no effect. If gamma stays 0, target network can be
deleted (~50% DQN params + sync logic). Not a consolidation item — flagged for research.

**What stays unchanged:**
- `TensorReplayBuffer`, `FusionRewardCalculator`, `fused_predict`, `QNetwork`, `Backbone`, `build_mlp_body`
- `MLPFusionModule`, `WeightedAvgModule` — already proper LightningModules

## Execution order

1. Dead code deletion — `fuse()` methods (standalone, no dependencies)
2. Registry dissolution — move fusion layout to `fusion_features.py`, class map to `_training.py`
3. Inline/move single-use utilities — `soft_label_kd_loss` → `gat.py`, `focal_loss` → `gat.py`
4. `temporal.py` — replace manual checkpoint unwrap with `load_inner_model`
5. `GraphModuleBase` in `_training.py` — shared `setup()`, OOM guard, metrics, threshold
6. Refactor VGAE/GAT/DGI to subclass `GraphModuleBase`, delete duplicated methods
7. Wire `add_optimizer_args` in `cli.py`, delete `configure_optimizers` from GAT/DGI/VGAE
8. `FusionModuleBase` — shared test/validate/predict for fusion agents
9. Refactor DQN → `DQNFusionModule(FusionModuleBase)`, delete `RLFusionModule`
10. Refactor Bandit → `BanditFusionModule(FusionModuleBase)`
11. Update YAML configs: `optimizer:` / `lr_scheduler:` sections, model class paths
12. Update `__init__.py` re-exports for new locations
13. Verify: import checks + `--collect-only` on test suite

## Line count estimate

| Section | Added | Deleted | Delta |
|---------|-------|---------|-------|
| Dead code (§4) | 0 | -14 | -14 |
| Registry dissolution (§5) | +25 | -85 | -60 |
| `_training.py` cleanup (§6) | +4 | -23 | -19 |
| `temporal.py` fix (§7) | +2 | -8 | -6 |
| `GraphModuleBase` + VGAE/GAT/DGI refactor (§2,3) | +40 | -90 | -50 |
| Optimizer wiring (§1) | +3 | -18 | -15 |
| DQN/Bandit conversion (§8) | +50 | -170 | -120 |
| **Total** | **+124** | **-408** | **-284** |

## Risks

- **Checkpoint compatibility:** Removing `configure_optimizers` changes optimizer state storage.
  Existing checkpoints load fine — Lightning stores optimizer state separately, and
  `load_from_checkpoint` doesn't require optimizer match.
- **Frozen teacher in optimizer:** If (a) causes measurable memory overhead on V100 16GB,
  switch to (b) — one shared override filtering `requires_grad` params.
- **binary_test_metrics callers:** 7 call sites (not 6). Fusion modules need the factory too —
  either keep as shared factory or inline in both `GraphModuleBase` and `FusionModuleBase`.
- **BinaryROC distributed:** torchmetrics handles DDP sync automatically. Single-GPU
  training (current setup) is unaffected.
- **VGAE configure_optimizers:** Not identical to GAT/DGI — needs projection param handling.
  Must verify CLI's `self.parameters()` includes projection before deleting.
