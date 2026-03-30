# Models Consolidation Plan

> Status: **complete** | Created: 2026-03-27 | Audited: 2026-03-30 | DQN/Bandit (§8): 2026-03-30 | §1-7,§11-13: 2026-03-30

Consolidate `graphids/core/models/` by adopting Lightning/torchmetrics
built-ins instead of hand-rolled shared logic. Three concerns: optimizers, setup, threshold.
Plus: registry dissolution, _training.py cleanup, DQN/Bandit Lightning conversion, temporal fix.

## 1. Optimizers — delete `configure_optimizers` from GAT/DGI, refactor VGAE ✅ DONE (2026-03-30)

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
- ~~`RLFusionModule` (`fusion_baselines.py:277-278`) — manual optimization, proxies agent's optimizer~~ (deleted — DQN/Bandit now have their own `configure_optimizers` as proper LightningModules)

YAML configs add `optimizer:` / `lr_scheduler:` sections instead of hardcoding in Python.

## 2. Setup — shared base class for lazy model construction ✅ DONE (2026-03-30)

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

## 3. Threshold — `BinaryROC` replaces manual accumulation ✅ DONE (2026-03-30)

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

## 4. Dead code deletion ✅ DONE (2026-03-30)

| What | Where | Why dead |
|------|-------|----------|
| `MLPFusionModule.fuse()` | `fusion_baselines.py:89-94` | References undefined `np` (numpy not imported), zero callers |
| `WeightedAvgModule.fuse()` | `fusion_baselines.py:161-168` | Same — `np` not imported, zero callers |

## 5. Registry dissolution ✅ DONE (2026-03-30)

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
    "fusion": "graphids.core.models.bandit.BanditFusionModule",  # was RLFusionModule (deleted)
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

## 6. `_training.py` cleanup ✅ DONE (2026-03-30)

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
`WeightedAvgModule`, `FusionModuleBase`) also call it, so keep the factory function
for fusion and inline in `GraphModuleBase` for core modules — or keep the factory for all.
(`RLFusionModule` deleted — `FusionModuleBase` now provides the shared base for fusion modules.)

## 7. `temporal.py` — use `load_inner_model` for GAT checkpoint loading ✅ DONE (2026-03-30)

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

## 8. DQN/Bandit — eliminate wrapper, adopt Lightning primitives ✅ DONE (2026-03-30)

~~`EnhancedDQNFusionAgent` (`dqn.py`) and `NeuralLinUCBAgent` (`bandit.py`) are plain Python
classes that re-implement Lightning concerns: optimizer (`dqn.py:156`), scheduler (`dqn.py:157`),
grad clipping (`dqn.py:260`, `bandit.py:223`), checkpointing (`dqn.py:349`, `bandit.py:345`).
`RLFusionModule` (`fusion_baselines.py:171-279`) wraps them with pure indirection.~~

**Completed.** What was done:

1. `FusionModuleBase(pl.LightningModule)` created in `fusion_baselines.py` — shared base with `test_step`, `validation_step`, abstract methods
2. `EnhancedDQNFusionAgent` → `DQNFusionModule(FusionModuleBase)` in `dqn.py` — proper LightningModule, auto-checkpoint, auto-device, `save_hparams`, `configure_optimizers`
3. `NeuralLinUCBAgent` → `BanditFusionModule(FusionModuleBase)` in `bandit.py` — same, plus `register_buffer` for `A_inv`, `b`, `theta`
4. `RLFusionModule` deleted from `fusion_baselines.py`
5. `fusion.yaml` updated to point to `BanditFusionModule`, `fusion_dqn.yaml` created for `DQNFusionModule`
6. `registry.py` updated with new lazy loaders
7. `reward_kwargs_from_cfg()` deleted from `fusion_reward.py` (dead code)
8. Backward-compat aliases: `EnhancedDQNFusionAgent = DQNFusionModule`, `NeuralLinUCBAgent = BanditFusionModule`

**Observation (still open):** gamma=0 is hardcoded (`dqn.py:119`), making `targets = rewards` unconditional
(`dqn.py:251-254`). Target network has no effect. If gamma stays 0, target network can be
deleted (~50% DQN params + sync logic). Not a consolidation item — flagged for research.

**What stayed unchanged:**
- `TensorReplayBuffer`, `FusionRewardCalculator`, `fused_predict`, `QNetwork`, `Backbone`, `build_mlp_body`
- `MLPFusionModule`, `WeightedAvgModule` — already proper LightningModules

## Execution order

All steps completed 2026-03-30:

1. ~~Dead code deletion — `fuse()` methods~~ ✅
2. ~~Registry dissolution — fusion layout to `fusion_features.py`, class map to `_training.py`~~ ✅
3. ~~Inline/move single-use utilities — `soft_label_kd_loss` → `gat.py`, `focal_loss` → `gat.py`~~ ✅
4. ~~`temporal.py` — replace manual checkpoint unwrap with `load_inner_model`~~ ✅
5. ~~`GraphModuleBase` in `_training.py` — shared `setup()`, OOM guard, metrics, threshold~~ ✅
6. ~~Refactor VGAE/GAT/DGI to subclass `GraphModuleBase`, delete duplicated methods~~ ✅
7. ~~Wire `add_optimizer_args` in `cli.py`, delete `configure_optimizers` from GAT/DGI~~ ✅
8. ~~`FusionModuleBase` — shared test/validate/predict for fusion agents~~ ✅
9. ~~Refactor DQN → `DQNFusionModule(FusionModuleBase)`, delete `RLFusionModule`~~ ✅
10. ~~Refactor Bandit → `BanditFusionModule(FusionModuleBase)`~~ ✅
11. ~~Update YAML configs: `optimizer:` / `lr_scheduler:` sections~~ ✅
12. ~~Update `__init__.py` re-exports for new locations~~ ✅
13. ~~Verify: import checks (all pass), MRO correct, fusion_state_dim=15~~ ✅

Full test suite via SLURM not yet run — import-level verification only.

## Line count (actual for §1-7,§11-13; estimated for §8)

Actual delta for §1-7,§11-13 (this session): **+131 / -121 = +10** across 9 files.
Less aggressive than estimated because `GraphModuleBase` added ~60 lines of new shared
infrastructure. The payoff is deduplication — 3 modules now share setup/OOM/threshold
instead of copying it.

| Section | Estimated | Actual |
|---------|-----------|--------|
| Dead code (§4) | -14 | -15 |
| Registry dissolution (§5) | -60 | done in prior commit (2c26e3f) |
| `_training.py` cleanup (§6) | -19 | ~-20 |
| `temporal.py` fix (§7) | -6 | -8 |
| `GraphModuleBase` + refactor (§2,3) | -50 | +10 (new base class offsets deletions) |
| Optimizer wiring (§1) | -15 | -8 (VGAE kept) |
| YAML configs (§11) | n/a | +18 |
| DQN/Bandit conversion (§8) | -120 (est.) | done in prior commit (4e34d16) |

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

## Completion notes (2026-03-30)

**What was done (§1-7, §11-13):**
- `GraphModuleBase` in `_training.py` — shared `setup()`, `_oom_safe_step()`, `_init_threshold_metrics()` + `_find_threshold()` (BinaryROC)
- VGAE, GAT, DGI all subclass `GraphModuleBase`. Duplicated `setup()` deleted from all three. `OOMSkipMixin` deleted.
- VGAE/DGI threshold: manual `_test_scores`/`_test_labels` list accumulation replaced with `roc_metric.update()` + `_find_threshold()`. Median fallback preserved for edge cases.
- `soft_label_kd_loss` inlined in `gat.py:_step`. `focal_loss` → `_focal_loss` in `gat.py`. Dead `soft_label_kd_loss` import removed from `vgae.py`.
- `temporal.py` checkpoint unwrap (10 lines) replaced with `load_inner_model("gat", ...)` (2 lines).
- `configure_optimizers` deleted from GAT and DGI. `cli.py` wires `add_optimizer_args(Adam)` + `add_lr_scheduler_args(CosineAnnealingLR)`. Stage YAMLs (`normal.yaml`, `curriculum.yaml`) have explicit `optimizer:` / `lr_scheduler:` sections.
- `binary_test_metrics` kept as shared factory (7 callers across core + fusion).
- `__init__.py` re-exports updated; `GraphModuleBase` added to public API.

**What was NOT done (deferred):**
- VGAE `configure_optimizers` kept — projection param handling needs verification.
- `lr` / `weight_decay` params in GAT/DGI `__init__` are now dead (saved to hparams but not read). Cleanup candidate.
- `T_max: 300` in YAMLs is static — old code used `self.trainer.max_epochs` dynamically. A `link_arguments` could fix this.
- No DGI stage YAML exists yet — DGI is placeholder in ablation recipe.
- DQN gamma=0 / target network deletion — flagged for research, not consolidation.
- Full SLURM test run not yet executed.
