# Artifacts Consolidation Plan

> Status: **proposed** | Date: 2026-03-27

Rewrite `graphids/core/artifacts/` (425 lines, 6 files) from dead standalone code into
Lightning-integrated callbacks and prediction writers. The package handles post-training
analysis (embeddings, CKA, loss landscape, fusion policy) — core to the research paper.

## Current state

**The entire package is dead code.** `generate_all` is exported but has zero callers.
All functions use manual inference loops; none use `Trainer.predict()` or `Trainer.test()`.
`fusion_policy.py` imports nonexistent `FusionResult`. Config references (`cfg.evaluation.*`,
`cfg.gat_stage`) don't exist in the post-LightningCLI schema.

| File | Lines | Status |
|------|-------|--------|
| `generate.py` | 64 | Dead orchestrator, zero callers |
| `embeddings.py` | 53 | Manual serialization of pre-computed results |
| `fusion_policy.py` | 31 | Broken import (`FusionResult` doesn't exist) |
| `cka.py` | 70 | Manual model loading + inference loops |
| `loss_landscape.py` | 204 | Manual model loading + inference loops |
| `__init__.py` | 3 | Re-exports dead function |

## Lightning integration architecture

### Core pattern: `Trainer.predict()` + `BasePredictionWriter` + callbacks

```bash
# Post-training artifact generation:
python -m graphids predict --ckpt_path best.ckpt --config predict.yaml

# Loss landscape after training:
python -m graphids fit --config train.yaml --loss_landscape true
```

Lightning handles model loading (`--ckpt_path`), device placement, distributed gathering,
and the predict loop. Artifacts plug in as callbacks configured via YAML.

### 1. Embeddings + Attention → `BasePredictionWriter`

Current `predict_step` returns `{errors, labels}` (VGAE) / `{preds, scores, labels}` (GAT).
For embeddings, a callback registers forward hooks on the inner model — no module changes.

```python
class EmbeddingWriter(BasePredictionWriter):
    """Captures embeddings via forward hooks, writes NPZ at epoch end."""

    def __init__(self, output_dir: str, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self._embeddings: list[torch.Tensor] = []
        self._hook_handles: list = []

    def on_predict_start(self, trainer, pl_module):
        handle = pl_module.model.register_forward_hook(self._capture)
        self._hook_handles.append(handle)

    def _capture(self, module, input, output):
        # Extract embedding from model's forward output
        ...

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        np.savez_compressed(self.output_dir / "embeddings.npz",
                            embeddings=torch.cat(self._embeddings).numpy())
        self._embeddings.clear()

    def on_predict_end(self, trainer, pl_module):
        for h in self._hook_handles:
            h.remove()
```

Configured via YAML — no code changes to run:
```yaml
predict:
  trainer:
    callbacks:
      - class_path: graphids.core.artifacts.EmbeddingWriter
        init_args:
          output_dir: artifacts/
```

### 2. CKA (teacher-student similarity) → `Callback`

CKA needs two models on the same data. A callback loads the teacher and uses forward hooks
on both models to collect per-layer intermediate representations.

```python
class CKACallback(Callback):
    """Compute CKA between student and teacher at predict time."""

    def __init__(self, teacher_ckpt_path: str, model_type: str = "gat",
                 max_samples: int = 500, output_dir: str = "."):
        self.teacher_ckpt_path = teacher_ckpt_path
        self.model_type = model_type
        self.max_samples = max_samples
        self.output_dir = Path(output_dir)
        self._student_reps: list[list[np.ndarray]] = []
        self._teacher_reps: list[list[np.ndarray]] = []

    def on_predict_start(self, trainer, pl_module):
        from graphids.core.models._training import load_inner_model
        self.teacher, _ = load_inner_model(
            self.model_type, self.teacher_ckpt_path, pl_module.device,
        )
        # Register hooks on both student and teacher intermediate layers
        ...

    def on_predict_epoch_end(self, trainer, pl_module, predictions):
        # Compute per-layer CKA scores from collected representations
        scores = [_linear_cka(s, t) for s, t in zip(student_layers, teacher_layers)]
        json.dump({"layer_cka": scores}, (self.output_dir / "cka.json").open("w"))
```

Domain math preserved as-is (20 lines):
- `_unbiased_hsic(K, L) -> float` — unbiased HSIC estimator
- `_linear_cka(X, Y) -> float` — linear CKA via unbiased HSIC

Configured via YAML:
```yaml
predict:
  trainer:
    callbacks:
      - class_path: graphids.core.artifacts.CKACallback
        init_args:
          teacher_ckpt_path: path/to/teacher.ckpt
          output_dir: artifacts/
```

### 3. Loss Landscape → `after_fit` hook on CLI

Loss landscape perturbs model weights across a grid — fundamentally different from predict
(which assumes fixed weights). Uses `Trainer.validate()` in a loop.

```python
class GraphIDSCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--loss_landscape", default=False)
        parser.add_argument("--loss_landscape.resolution", type=int, default=51)
        parser.add_argument("--loss_landscape.scale", type=float, default=1.0)
        # ...existing link_arguments...

    def after_fit(self):
        if self.config.get("loss_landscape"):
            from graphids.core.artifacts.loss_landscape import sweep_loss_landscape
            sweep_loss_landscape(
                self.model, self.datamodule, self.trainer,
                resolution=self.config.loss_landscape.resolution,
                scale=self.config.loss_landscape.scale,
            )
```

The sweep function reuses `Trainer.validate()` for each grid point:
```python
def sweep_loss_landscape(model, datamodule, trainer, resolution, scale, seed=42):
    base_params = [p.clone() for p in model.model.parameters()]
    dir1 = _random_direction(model.model, seed)
    dir2 = _random_direction(model.model, seed + 1)
    results = []
    for alpha in linspace(-scale, scale, resolution):
        for beta in linspace(-scale, scale, resolution):
            _perturb_model(model.model, base_params, dir1, dir2, alpha, beta)
            metrics = trainer.validate(model, datamodule, verbose=False)
            results.append({"alpha": alpha, "beta": beta, "loss": metrics[0]["val_loss"]})
    # Restore original weights
    for p, bp in zip(model.model.parameters(), base_params):
        p.data.copy_(bp)
    _save_parquet(results, ...)
```

Domain logic preserved (~50 lines):
- `_filter_normalize(direction, reference)` — Li et al. filter normalization
- `_random_direction(model, seed)` — generates one filter-normalized random direction
- `_perturb_model(model, base_params, dir1, dir2, alpha, beta)` — in-place perturbation

Evidence: Lightning docs `cli/lightning_cli_expert.rst` — `after_fit` hook for post-training logic.

### 4. Fusion Policy → `BasePredictionWriter`

DQN/Bandit `predict_step` already returns `{preds, fused_scores, alphas}`.
A writer serializes Q-values and alpha distributions.

```python
class FusionPolicyWriter(BasePredictionWriter):
    """Write fusion policy (alphas, Q-values) to JSON after predict."""

    def __init__(self, output_dir: str, write_interval: str = "epoch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        all_alphas = torch.cat([p["alphas"] for p in predictions])
        all_scores = torch.cat([p["fused_scores"] for p in predictions])
        policy = {
            "alpha_mean": float(all_alphas.mean()),
            "alpha_std": float(all_alphas.std()),
            "score_distribution": all_scores.tolist(),
        }
        json.dump(policy, (self.output_dir / "fusion_policy.json").open("w"), indent=2)
```

## Proposed file layout

```
graphids/core/artifacts/
  __init__.py              # re-exports: EmbeddingWriter, CKACallback,
                           #   FusionPolicyWriter, sweep_loss_landscape
  writers.py               # EmbeddingWriter, FusionPolicyWriter
                           #   (BasePredictionWriter subclasses, ~60 lines)
  cka.py                   # CKACallback + _unbiased_hsic, _linear_cka (~50 lines)
  loss_landscape.py        # sweep_loss_landscape + domain math (~80 lines)
```

## What's preserved vs deleted

| Current code | Preserved? | New location |
|---|---|---|
| `_unbiased_hsic`, `_linear_cka` (20 lines) | Yes | `cka.py` |
| `_filter_normalize`, `_random_direction`, `_perturb_model` (50 lines) | Yes | `loss_landscape.py` |
| Manual model loading in `cka.py`, `loss_landscape.py` (~30 lines) | **Deleted** | Lightning handles via `--ckpt_path` / callback hooks |
| Manual inference loops: `_collect_reps`, `_vgae_loss`, `_gat_loss`, `_sweep_grid` (~100 lines) | **Deleted** | `Trainer.predict()` / `Trainer.validate()` |
| `generate_all` orchestrator (64 lines) | **Deleted** | Callbacks self-register via YAML config |
| `embeddings.py` (53 lines) | Rewritten | `EmbeddingWriter` in `writers.py` (~30 lines) |
| `fusion_policy.py` (31 lines, broken import) | Rewritten | `FusionPolicyWriter` in `writers.py` (~25 lines) |

## Execution order

1. Delete all current files (dead code, broken imports, stale config refs)
2. Create `writers.py` — `EmbeddingWriter` + `FusionPolicyWriter`
3. Rewrite `cka.py` — `CKACallback` wrapping preserved domain math
4. Rewrite `loss_landscape.py` — `sweep_loss_landscape` using `Trainer.validate()`
5. Wire `after_fit` hook in `__main__.py` for loss landscape
6. Update `__init__.py` re-exports
7. Add YAML examples for predict-time artifact generation
8. Verify: `python -m graphids predict --help` shows callback args

## Line count

| Change | Lines |
|--------|-------|
| Delete current package | -425 |
| `writers.py` (EmbeddingWriter + FusionPolicyWriter) | +60 |
| `cka.py` (CKACallback + domain math) | +50 |
| `loss_landscape.py` (sweep + domain math) | +80 |
| `__init__.py` + CLI wiring | +15 |
| **Net** | **-220 lines** |

## Risks

- **Embedding hook design:** The forward hook approach depends on the inner model's
  architecture exposing useful intermediate tensors. VGAE's encoder output (`z`) and GAT's
  pre-pool embeddings are the targets. Need to verify hook attachment points per model.
- **Loss landscape memory:** `Trainer.validate()` in a loop (resolution² calls) may be
  slower than the current manual loop due to Trainer overhead per call. If so, can batch
  multiple perturbations or use a raw `model.eval()` loop as fallback within the sweep
  function (keeping `Trainer` for the outer checkpoint/device setup only).
- **CKA teacher loading:** The callback loads the teacher in `on_predict_start`, which means
  both student and teacher are on the same device simultaneously. On V100 16GB this is fine
  for GAT (~50MB per model) but verify for VGAE if latent dims are large.
