# Models Consolidation Plan

> Status: **proposed** | Date: 2026-03-27

Consolidate `graphids/core/models/` (3,232 lines, 13 files) by adopting Lightning/torchmetrics
built-ins instead of hand-rolled shared logic. Three concerns: optimizers, setup, threshold.

## 1. Optimizers — delete `configure_optimizers` from VGAE/GAT/DGI

LightningCLI creates optimizers automatically from `--optimizer` / `--lr_scheduler` CLI args
when the module doesn't define `configure_optimizers`. The current `__main__.py` already uses
`LightningCLI` with `subclass_mode_model=True`.

**Action:** Pin defaults in `GraphIDSCLI.add_arguments_to_parser`:

```python
parser.add_optimizer_args(torch.optim.Adam)
parser.add_lr_scheduler_args(torch.optim.lr_scheduler.CosineAnnealingLR)
```

Then delete identical `configure_optimizers` from:
- `vgae.py:511-517`
- `gat.py:290-293`
- `dgi.py:269-272`

YAML configs specify `optimizer: {lr: 0.001, weight_decay: 1e-5}` and
`lr_scheduler: {T_max: 100}` instead of hardcoding in Python.

**Frozen teacher params:** CLI uses `self.parameters()` which includes the frozen teacher
(requires_grad=False). Two options:
- (a) Accept tiny overhead — frozen params get Adam state but no grad computation
- (b) Override `configure_optimizers` once in a shared base to filter `requires_grad` params

Recommend (a) for simplicity unless profiling shows meaningful memory cost.

**Exceptions that keep custom `configure_optimizers`:**
- `TemporalLightningModule` — two param groups with different LRs (spatial × temporal)
- `RLFusionModule` — manual optimization, returns agent's optimizer directly

Evidence: Lightning docs `cli/lightning_cli_intermediate_2.rst` (optimizer from CLI),
`cli/lightning_cli_advanced_3.rst` (`add_optimizer_args` pattern).

## 2. Setup — shared base class for lazy model construction

VGAE, GAT, DGI all copy-paste an identical `setup()` method (6 lines each):

```python
def setup(self, stage=None):
    if self.model is None:
        dm = self.trainer.datamodule
        self.hparams.num_ids = dm.num_ids
        self.hparams.in_channels = dm.in_channels
        self.hparams.num_classes = dm.num_classes
        self._build()
```

This is Lightning-idiomatic — no deeper framework solution exists. A shared base class
`GraphModuleBase(OOMSkipMixin, pl.LightningModule)` provides this once. Each subclass
defines `_build()` only.

**Action:** Create `GraphModuleBase` in `_training.py` (no new file needed):

```python
class GraphModuleBase(OOMSkipMixin, pl.LightningModule):
    """Shared base for VGAE, GAT, DGI modules."""

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            self.hparams.num_ids = dm.num_ids
            self.hparams.in_channels = dm.in_channels
            self.hparams.num_classes = dm.num_classes
            self._build()

    def _build(self):
        raise NotImplementedError
```

Then `VGAEModule(GraphModuleBase)`, `GATModule(GraphModuleBase)`, `DGIModule(GraphModuleBase)`.

## 3. Threshold — `BinaryROC` metric replaces manual accumulation

VGAE and DGI both maintain manual `_test_scores`/`_test_labels` lists, concatenate at epoch
end, then run Youden-J threshold selection. This is 50 lines of character-for-character
identical code across the two modules.

The torchmetrics class-based `BinaryROC()` accumulates across batches automatically,
eliminating the manual lists.

**Action:** Add `BinaryROC` metric + shared `_find_threshold()` method to `GraphModuleBase`:

```python
from torchmetrics.classification import BinaryROC

class GraphModuleBase(OOMSkipMixin, pl.LightningModule):
    # ...setup from above...

    def _init_threshold_test(self):
        """Call from __init__ for unsupervised modules (VGAE, DGI)."""
        self.test_threshold: float | None = None
        self.roc_metric = BinaryROC()

    def _find_threshold(self):
        """Youden-J optimal threshold from accumulated ROC data."""
        fpr, tpr, thresholds = self.roc_metric.compute()
        if thresholds.numel() < 2:
            return None
        j = tpr - fpr
        best = torch.argmax(j)
        return float(thresholds[best]) if best < len(thresholds) else None

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "test_threshold") and self.test_threshold is not None:
            checkpoint["test_threshold"] = self.test_threshold

    def on_load_checkpoint(self, checkpoint):
        if "test_threshold" in checkpoint:
            self.test_threshold = checkpoint.get("test_threshold")
```

VGAE/DGI `test_step` simplifies to:
```python
def test_step(self, batch, _idx):
    scores = self._per_graph_errors(batch)  # or _per_graph_scores
    self.roc_metric.update(scores, batch.y)
    self.test_metrics.update((scores >= self.test_threshold).long(), batch.y)
```

Deletes: `_test_scores`, `_test_labels`, all `.append()`, `.clear()`, `torch.cat()` calls.

GAT and Temporal don't use threshold (supervised — torchmetrics directly). They inherit
`GraphModuleBase` but don't call `_init_threshold_test()`.

## 4. Dead code deletion

| What | Where | Why dead |
|------|-------|----------|
| `MLPFusionModule.fuse()` | `fusion_baselines.py:86-91` | References undefined `np`, zero callers |
| `WeightedAvgModule.fuse()` | `fusion_baselines.py:154-161` | Same — `np` not imported, zero callers |
| `FusionResult` import | `fusion_policy.py:10` | Class doesn't exist (deleted with eval_types.py) |
| `FusionResult` docstring | `generate.py:30` | Stale reference |

## Execution order

1. Dead code deletion (standalone, no dependencies)
2. `GraphModuleBase` in `_training.py` with shared `setup()` + threshold helpers
3. Refactor VGAE/GAT/DGI to subclass `GraphModuleBase`, delete duplicated methods
4. Wire `add_optimizer_args` in `__main__.py`, delete `configure_optimizers` from 3 modules
5. Update YAML configs to include `optimizer:` / `lr_scheduler:` sections
6. Verify: `python -c "from graphids.core.models.vgae import VGAEModule"` + `--collect-only`

## Line count estimate

| Change | Added | Deleted |
|--------|-------|---------|
| `GraphModuleBase` in `_training.py` | +35 | 0 |
| VGAE refactor | +2 | -35 |
| GAT refactor | +2 | -20 |
| DGI refactor | +2 | -35 |
| `__main__.py` optimizer wiring | +3 | 0 |
| Dead code deletion | 0 | -20 |
| **Net** | **+44** | **-110** |
| **Delta** | | **-66 lines** |

## 5. DQN/Bandit — eliminate wrapper, adopt Lightning primitives

### Problem

`EnhancedDQNFusionAgent` and `NeuralLinUCBAgent` are plain Python classes that manage their
own optimizers, grad clipping, schedulers, and checkpointing — all things Lightning handles
natively. `RLFusionModule` (`fusion_baselines.py:164-241`) wraps them with pure indirection:
every method delegates to the agent.

### What Lightning already provides (re-implemented by agents)

| Concern | Current (manual) | Lightning equivalent |
|---------|-----------------|---------------------|
| Optimizer | `self.optimizer = AdamW(...)` (`dqn.py:168`) | `configure_optimizers` return |
| Scheduler | `self.scheduler = ReduceLROnPlateau(...)` (`dqn.py:169`) | `configure_optimizers` return |
| Grad clipping | `clip_grad_norm_` (`dqn.py:272`, `bandit.py:223`) | `Trainer(gradient_clip_val=1.0)` or `self.clip_gradients()` |
| Checkpoint save | Custom `state_dict()` (`dqn.py:361-367`, `bandit.py:345-355`) | Auto-saves `nn.Module` attrs + `register_buffer` tensors |
| Checkpoint load | Custom `load_checkpoint()` (`dqn.py:369-378`, `bandit.py:357-369`) | `load_from_checkpoint()` built-in |
| Device transfer | Manual `.to(device)` everywhere | Lightning auto device placement |

### Action: make DQN/Bandit LightningModules, delete RLFusionModule

**DQNFusionModule(FusionModuleBase):**
- `q_network`, `target_network` as `nn.Module` attributes → auto-checkpointed
- `automatic_optimization = False` (multiple gradient steps per `training_step`)
- `configure_optimizers` returns `AdamW` + `ReduceLROnPlateau`
- Target network update → `on_train_batch_end` hook
- Epsilon decay → end of `training_step` (stays inline)
- Only `epsilon` needs `on_save_checkpoint` (scalar)
- Delete `state_dict()`, `load_checkpoint()`, manual optimizer creation

**BanditFusionModule(FusionModuleBase):**
- `backbone` as `nn.Module` attribute → auto-checkpointed
- `A_inv`, `b`, `theta` → `register_buffer()` → auto-checkpointed + auto device transfer
- `configure_optimizers` returns `AdamW`
- Delete `state_dict()`, `load_checkpoint()`, manual `.to(device)` calls

**FusionModuleBase(pl.LightningModule):**
- `automatic_optimization = False`
- `reward_calc`, `decision_threshold`, `alpha_values`
- `test_step` / `on_test_epoch_start` / `on_test_epoch_end` (identical in both)
- `predict()` → `fused_predict()` (already factored in `fusion_reward.py`)
- `validate_batch()` pattern (nearly identical: `dqn.py:334-355` vs `bandit.py:312-325`)

### Observation: target network is dead with gamma=0

`dqn.py:264-266`: gamma=0 means `targets = rewards` — no bootstrapping from next state.
The target network (`dqn.py:164`, synced at `dqn.py:276-277`) has no effect. If gamma stays
0, the target network can be deleted (saves ~50% of DQN parameters + the sync logic).
Not a consolidation item — flagged for research decision.

### What stays unchanged

- `TensorReplayBuffer` — no framework replacement, clean implementation
- `FusionRewardCalculator` / `fused_predict` — domain logic, already well-factored
- `QNetwork` / `Backbone` as `nn.Module` — correct as-is
- `build_mlp_body` — shared utility
- `MLPFusionModule` / `WeightedAvgModule` — already proper LightningModules

### Line count estimate (DQN/Bandit section)

| Change | Added | Deleted |
|--------|-------|---------|
| `FusionModuleBase` in `_training.py` or `fusion_baselines.py` | +30 | 0 |
| `DQNFusionModule` replaces `EnhancedDQNFusionAgent` | +10 | -50 |
| `BanditFusionModule` replaces `NeuralLinUCBAgent` | +10 | -40 |
| Delete `RLFusionModule` | 0 | -80 |
| **Section net** | **+50** | **-170** |
| **Section delta** | | **-120 lines** |

## 6. Registry — split into LightningCLI class resolution + fusion layout

### Problem

`registry.py` (104 lines) couples two unrelated concerns in a single `_MODELS` dict:
1. Model class resolution (`get`, `get_module_cls`) — a LightningCLI feature
2. Fusion state layout (`fusion_state_dim`, `feature_layout`, `extractors`) — domain logic

### Current registry contents

```python
_MODELS = {
    "vgae": (GraphAutoencoderNeighborhood.from_config, VGAEFusionExtractor(), _vgae_module),
    "gat":  (GATWithJK.from_config,                    GATFusionExtractor(),  _gat_module),
    "dqn":  (_dqn_from_config,                          None,                  None),
    "dgi":  (_dgi_from_config,                          None,                  _dgi_module),
}
```

Each entry bundles `(arch_factory, fusion_extractor, module_class_loader)`.

### Callers

| Function | Callers | Status |
|----------|---------|--------|
| `get()` (arch factory) | **Zero** — re-exported in `__init__.py` but never called | Dead |
| `get_module_cls()` | 1 — `load_inner_model` (`_training.py:158`) | Replaceable |
| `fusion_state_dim()` | `QNetwork.from_config`, `DQN.from_config`, `Bandit.from_config` | Domain logic, stays |
| `feature_layout()` | `FusionRewardCalculator`, `WeightedAvgModule` | Domain logic, stays |
| `extractors()` | `datamodule.py:251` fusion feature extraction | Domain logic, stays |

### Action

**Delete `get()`** — zero callers, dead code.

**Replace `get_module_cls()`** — single caller is `load_inner_model`, which maps a
`model_type` string → LightningModule class for `load_from_checkpoint`. With LightningCLI,
model classes are resolved from `class_path` strings in YAML config. Replace with:

```python
_MODULE_PATHS: dict[str, str] = {
    "vgae": "graphids.core.models.vgae.VGAEModule",
    "gat": "graphids.core.models.gat.GATModule",
    "dgi": "graphids.core.models.dgi.DGIModule",
}

def load_inner_model(model_type, ckpt_path, device):
    import importlib
    module_path, cls_name = _MODULE_PATHS[model_type].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    module = cls.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=True)
    ...
```

This is a simple dict in `_training.py` next to `load_inner_model` — no registry indirection,
no lazy loader functions (`_vgae_module`, `_gat_module`, `_dgi_module`).

**Move fusion layout functions to `fusion_features.py`** — this file already defines the
extractor Protocol and implementations (`VGAEFusionExtractor`, `GATFusionExtractor`).
Move `FeatureLayout`, `fusion_state_dim()`, `feature_layout()`, `extractors()` there.
The registration-order coupling becomes explicit:

```python
# fusion_features.py
_EXTRACTORS: list[tuple[str, FusionFeatureExtractor]] = [
    ("vgae", VGAEFusionExtractor()),
    ("gat", GATFusionExtractor()),
]
# Order is load-bearing — matches 15-D state layout that trained DQN checkpoints expect.
```

**Delete `registry.py`** — nothing remains.

**Update `__init__.py` re-exports** — point at new locations in `fusion_features.py` and
`_training.py`.

### Line count estimate (registry section)

| Change | Added | Deleted |
|--------|-------|---------|
| Move layout functions to `fusion_features.py` | +25 | 0 |
| Inline class map in `_training.py` | +8 | 0 |
| Delete `registry.py` | 0 | -104 |
| Update `__init__.py` re-exports | +2 | -4 |
| Delete lazy loaders + `get()` + `get_module_cls()` | 0 | (in registry.py total) |
| **Section net** | **+35** | **-108** |
| **Section delta** | | **-73 lines** |

## 7. `_training.py` — inline framework wrappers, keep domain logic

### Problem

`_training.py` (207 lines, 8 functions) mixes domain utilities with thin wrappers around
Lightning/PyTorch that should be inlined at call sites or folded into base classes.

### Audit

| Function | Lines | Callers | Verdict |
|----------|-------|---------|---------|
| `compute_node_budget()` | 43 | `datamodule.py`, `curriculum.py`, `vgae.py` | **Keep** — domain logic (VRAM × dataset stats) |
| `teacher_on_device()` | 14 | VGAE `_step`, GAT `_step` | **Keep** — KD CPU-offload, 2 callers |
| `load_inner_model()` | 22 | `prepare_kd`, `datamodule.py`, `curriculum.py`, `cka.py` | **Keep** — thin but 5 callers across packages |
| `prepare_kd()` | 41 | VGAE `_build`, GAT `_build` | **Keep** — KD teacher resolution |
| `soft_label_kd_loss()` | 6 | GAT `_step` (1 caller) | **Inline** — `F.kl_div(log_softmax(s/T), softmax(t/T)) * T²`, 4 lines |
| `focal_loss()` | 4 | GAT `__init__` (1 caller) | **Move to `gat.py`** — private 3-line helper, not shared |
| `binary_test_metrics()` | 11 | 6 callers → 1 after base class | **Inline in `GraphModuleBase.__init__`** — just `MetricCollection(...)` construction |
| `OOMSkipMixin` | 11 | VGAE, GAT, DGI → base class | **Fold into `GraphModuleBase`** — mixin has no independent value |

### Actions

**Inline `soft_label_kd_loss`** at its single call site (`gat.py:254`):

```python
# gat.py _step, replacing: kd_loss = soft_label_kd_loss(logits, t_logits, kd.temperature)
T = kd.temperature
kd_loss = F.kl_div(
    F.log_softmax(logits / T, dim=-1),
    F.softmax(t_logits / T, dim=-1),
    reduction="batchmean",
) * T ** 2
```

**Move `focal_loss` to `gat.py`** as a module-private `_focal_loss`. Only GATModule uses it
(`gat.py:219`). Not worth a shared utility for 1 consumer.

**Inline `binary_test_metrics` into `GraphModuleBase.__init__`:**

```python
class GraphModuleBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.test_metrics = MetricCollection({
            "accuracy": BinaryAccuracy(), "f1": BinaryF1Score(),
            "precision": BinaryPrecision(), "recall": BinaryRecall(),
            "specificity": BinarySpecificity(), "auc": BinaryAUROC(),
        })
```

Fusion modules (`MLPFusionModule`, `WeightedAvgModule`, `RLFusionModule` → `FusionModuleBase`)
do the same inline — or `FusionModuleBase` inherits the same pattern.

**Fold `OOMSkipMixin` into `GraphModuleBase`:**

```python
class GraphModuleBase(pl.LightningModule):
    def _oom_safe_step(self, batch, batch_idx, step_fn):
        try:
            return step_fn(batch, batch_idx)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            _log.warning("oom_batch_skipped", batch_idx=batch_idx,
                         num_graphs=batch.num_graphs, num_nodes=batch.num_nodes)
            return None
```

### Resulting `_training.py` contents

After cleanup, `_training.py` contains only domain utilities:

```
_training.py (~120 lines)
├── NodeBudgetInfo + compute_node_budget()   # VRAM-aware batch sizing
├── teacher_on_device()                       # KD teacher CPU offload
├── load_inner_model()                        # checkpoint → inner nn.Module
├── prepare_kd()                              # teacher resolution + projection
└── GraphModuleBase                           # shared base (setup, OOM, metrics, threshold)
```

### Line count estimate (`_training.py` section)

| Change | Added | Deleted |
|--------|-------|---------|
| Inline `soft_label_kd_loss` in `gat.py` | +4 | -6 |
| Move `focal_loss` to `gat.py` | +4 | -4 |
| Inline `binary_test_metrics` in base classes | +6 | -11 |
| Fold `OOMSkipMixin` into `GraphModuleBase` | 0 | -11 |
| **Section net** | **+14** | **-32** |
| **Section delta** | | **-18 lines** |

## 8. `temporal.py` — use `load_inner_model` for GAT checkpoint loading

### Problem

`TemporalLightningModule._build_model` (`temporal.py:211-218`) manually unwraps a Lightning
checkpoint, strips `"model."` prefixes, and loads into the inner GAT nn.Module:

```python
checkpoint = torch.load(gat_ckpt_path, map_location="cpu", weights_only=True)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    raw = checkpoint["state_dict"]
    checkpoint = {k.replace("model.", ""): v for k, v in raw.items() if k.startswith("model.")}
gat.load_state_dict(checkpoint)
gat.eval()
```

This is exactly what `load_inner_model("gat", path, "cpu")` in `_training.py` already does
via `load_from_checkpoint` + `.model` extraction.

### Action

Replace the manual unwrapping with:

```python
from ._training import load_inner_model
gat, _ = load_inner_model("gat", gat_ckpt_path, "cpu")
```

Delete the manual `torch.load` + `state_dict` prefix stripping + `load_state_dict` + `eval()`.
Also removes the now-unnecessary `gat = GATWithJK.from_config(cfg, ...)` construction on the
training path (line 210) since `load_inner_model` handles construction internally.

The reconstruction path (no checkpoint, `load_from_checkpoint` flow) still needs a skeleton
GAT — keep `GATWithJK.from_config` for that branch only.

### Not changed

- `configure_optimizers` — legitimately custom (two param groups, spatial LR factor). Stays.
- `_shared_step` device handling — 1-line pattern repeated 3x, not worth extracting.
- No `GraphModuleBase` subclassing — eager build in `__init__`, no OOM guard, no threshold.
- `TemporalGraphClassifier` nn.Module — pure architecture (PyTorch TransformerEncoder).
- `_conv.py` (224 lines) — pure PyG building blocks, no Lightning overlap. No changes.

### Line count estimate (temporal section)

| Change | Added | Deleted |
|--------|-------|---------|
| Replace manual checkpoint load with `load_inner_model` | +2 | -12 |
| **Section delta** | | **-10 lines** |

## Updated execution order

1. Dead code deletion (standalone, no dependencies)
2. Dissolve `registry.py` — move fusion layout to `fusion_features.py`, class map to `_training.py`
3. Inline/move single-use utilities: `soft_label_kd_loss` → `gat.py`, `focal_loss` → `gat.py`, delete `OOMSkipMixin` class
4. `temporal.py` — replace manual checkpoint unwrapping with `load_inner_model`
5. `GraphModuleBase` in `_training.py` — shared `setup()`, OOM guard, metrics, threshold helpers
6. Refactor VGAE/GAT/DGI to subclass `GraphModuleBase`, delete duplicated methods
7. Wire `add_optimizer_args` in `__main__.py`, delete `configure_optimizers` from 3 modules
8. `FusionModuleBase` — shared test/validate/predict for fusion agents
9. Refactor DQN → `DQNFusionModule(FusionModuleBase)`, delete `RLFusionModule`
10. Refactor Bandit → `BanditFusionModule(FusionModuleBase)`
11. Update YAML configs: `optimizer:` / `lr_scheduler:` sections, model class paths
12. Update `__init__.py` re-exports for new locations
13. Verify: import checks + `--collect-only` on test suite

## Combined line count

| Section | Delta |
|---------|-------|
| Dead code + VGAE/GAT/DGI (sections 1, 5-7) | -66 |
| DQN/Bandit (sections 8-10) | -120 |
| Registry dissolution (section 2) | -73 |
| `_training.py` cleanup (section 3) | -18 |
| `temporal.py` fix (section 4) | -10 |
| **Total** | **-287 lines** |

## Risks

- **Checkpoint compatibility:** Removing `configure_optimizers` changes how optimizer state
  is stored. Existing checkpoints load fine (Lightning stores optimizer state separately from
  model weights, and `load_from_checkpoint` doesn't require optimizer match).
- **Frozen teacher in optimizer:** If (a) causes measurable memory overhead on V100 16GB,
  switch to (b) — one shared override filtering `requires_grad` params.
- **BinaryROC distributed:** torchmetrics handles DDP sync automatically. Single-GPU
  training (current setup) is unaffected.
