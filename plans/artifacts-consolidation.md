# Artifacts Consolidation Plan

> Status: **superseded** | Original: 2026-03-27 | Updated: 2026-03-28

## What happened

This plan proposed rewriting `graphids/core/artifacts/` using `BasePredictionWriter` callbacks,
a `CKACallback`, and a `sweep_loss_landscape` wired to `LightningCLI.after_fit`.

**The actual implementation took a different approach:** the `Analyzer` subcommand
(`python -m graphids analyze`) was built instead. It uses jsonargparse directly (no Trainer),
loads checkpoints, and generates artifacts as standalone operations.

See commits `dc90f2d` and `6928e03` (2026-03-28).

## Current state of `graphids/core/artifacts/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 127 | Exports `Analyzer` class |
| `analyzer.py` | ~200 | Post-training artifact generation (embeddings, CKA, loss landscape) |
| `cka.py` | ~65 | CKA analysis (domain math preserved from original plan) |
| `embeddings.py` | ~110 | Embedding extraction |
| `fusion_policy.py` | ~30 | Fusion policy serialization |
| `loss_landscape.py` | ~200 | Loss landscape analysis |

## CLI usage

```bash
python -m graphids analyze --config graphids/config/stages/analyze_vgae.yaml \
    --analyzer.ckpt_path path/to/best.ckpt --analyzer.dataset hcrl_sa
```

YAML keys nest under `analyzer:`. Required args (`ckpt_path`, `dataset`, `model_type`) have
no defaults -- jsonargparse rejects configs that omit them. Fail-loud on missing checkpoints.

## What was preserved from this plan

- Domain math: `_unbiased_hsic`, `_linear_cka` (CKA)
- Domain math: `_filter_normalize`, `_random_direction`, `_perturb_model` (loss landscape)
- Dead `generate_all` orchestrator was deleted
- Broken `FusionResult` import was fixed

## What differed from this plan

| This plan proposed | What was built instead |
|---|---|
| `BasePredictionWriter` subclasses | `Analyzer` class with direct model loading |
| `CKACallback` on `Trainer.predict()` | `analyzer.py` runs CKA standalone |
| `sweep_loss_landscape` on `LightningCLI.after_fit` | `analyzer.py` runs landscape standalone |
| `predict` YAML configs with callback lists | `analyze` YAML configs under `analyzer:` namespace |

## Original plan (archived below)

The rest of this file preserves the original proposed design for historical reference.
The callback-based approach remains a valid alternative if artifacts need to run as part
of the training loop in the future.

---

<details>
<summary>Original proposed design (2026-03-27)</summary>

### Lightning integration architecture

#### Core pattern: `Trainer.predict()` + `BasePredictionWriter` + callbacks

```bash
# Post-training artifact generation:
python -m graphids predict --ckpt_path best.ckpt --config predict.yaml

# Loss landscape after training:
python -m graphids fit --config train.yaml --loss_landscape true
```

#### 1. Embeddings + Attention -> `BasePredictionWriter`

Forward hooks on inner model to capture embeddings during predict loop.

#### 2. CKA -> `Callback`

Loads teacher in `on_predict_start`, registers hooks on both models.

#### 3. Loss Landscape -> `after_fit` hook on CLI

Weight perturbation grid using `Trainer.validate()` in a loop.

#### 4. Fusion Policy -> `BasePredictionWriter`

Serializes Q-values and alpha distributions from predict_step output.

</details>
