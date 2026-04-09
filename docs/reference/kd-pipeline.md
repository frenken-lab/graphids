# Knowledge Distillation Pipeline

> Status: Wired but untested end-to-end | See frenken-lab/graphids#25

## What KD does

Large ("teacher") models are trained first; their knowledge is compressed into small
("student") models via auxiliary loss terms injected at build time.

- **GAT KD** (`SoftLabelDistillation`): `a * KL(student/T || teacher/T) * T^2 + (1-a) * task_loss`
- **VGAE KD** (`FeatureDistillation`): `a * (latent_w * MSE(z_s, z_t) + recon_w * MSE(cont_s, cont_t)) + (1-a) * task_loss`

Both loss classes live in `graphids/core/losses/distillation.py`. The teacher is held on
CPU via `__dict__` assignment (bypassing `nn.Module` registration so Lightning doesn't
auto-transfer it) and moved to the student device only during `forward` via
`_run_teacher_on()` (`distillation.py:52`).

## How to enable KD in a recipe

Add a `kd:` block (a `KDEntry`) to the sweep entry and set `teacher_config` to the name
of the recipe config that produces the teacher checkpoint:

```yaml
sweeps:
  - model_family: gat
    scale: small
    kd:
      alpha: 0.7
      teacher_config: gat_large          # REQUIRED: names another recipe config
      temperature: 4.0                   # GAT only
```

`teacher_config` is planning-only: `enumerate_assets` (`planner.py:107`) resolves it to
an upstream asset dependency so the teacher trains first. The resolved checkpoint path
flows through as `kd_overrides["teacher_ckpt"]` on `StageConfig` and is passed to the
stage as the `distillation_config` TLA.

## Data flow

```
Recipe kd: block (KDEntry fields: type, alpha, teacher_config, teacher_scale,
  temperature, model_path, vgae_latent_weight, vgae_recon_weight)
  |
  v
enumerate_assets (planner.py:107)
  - Sets kd_tag="_kd" on StageConfig (distinct identity hash)
  - Post-pass: looks up teacher asset by teacher_config name,
    wires as upstream_asset_names. Raises on missing asset.
  - Stores raw payload in StageConfig.kd_overrides (planner.py:41)
  |
  v
ResolvedConfig.resolve (resolve.py)
  kd_overrides -> jsonnet_tla["distillation_config"] (resolve.py:60-61)
  |
  v
render_config(jsonnet_path, jsonnet_tla)  (config/jsonnet.py)
  Stage jsonnet emits model.init_args.distillation_config when non-null
  |
  v
validate_config(rendered)  (config/schemas.py)
  |
  v
graphids.instantiate.instantiate (instantiate.py)
  inject_loss_fn() (losses/build.py:105) pops distillation_config from
  init_args, calls build_loss() which:
    - Loads teacher checkpoint via load_inner_model() (base.py:238)
    - Wraps base loss in SoftLabelDistillation (GAT) or
      FeatureDistillation (VGAE, with optional nn.Linear projection
      if latent dims differ)
    - Injects result as loss_fn= kwarg to the model constructor
  |
  v
Model.__init__(loss_fn=SoftLabelDistillation | FeatureDistillation)
  Model stores loss_fn; save_hyperparameters(ignore=["loss_fn"])
  |
  v
Model.training_step -> self.loss_fn(outputs, batch)
  Loss module handles teacher forward + KD math internally.
  last_hard_loss / last_soft_loss populated after each forward
  so LightningModule can log components.
```

## Key code

| Symbol | Location | Role |
|--------|----------|------|
| `KDEntry` | `graphids/orchestrate/planning/recipes.py:54` | Recipe-side Pydantic schema |
| `SoftLabelDistillation` | `graphids/core/losses/distillation.py:62` | Hinton soft-label KD loss (GAT) |
| `FeatureDistillation` | `graphids/core/losses/distillation.py:123` | Feature-based KD loss (VGAE) |
| `_attach_teacher` | `graphids/core/losses/distillation.py:38` | Parks teacher in `__dict__`, bypasses Lightning auto-transfer |
| `_run_teacher_on` | `graphids/core/losses/distillation.py:52` | Move teacher to device, run under `no_grad`, move back |
| `build_loss` | `graphids/core/losses/build.py:25` | Builds base loss + optional KD wrapper from config dicts |
| `inject_loss_fn` | `graphids/core/losses/build.py:105` | Pops loss/distillation config from init_args, injects `loss_fn` |
| `kd_overrides` | `graphids/orchestrate/planning/planner.py:41` | `StageConfig` field carrying the KD payload dict |
| `enumerate_assets` | `graphids/orchestrate/planning/planner.py:107` | Wires teacher asset as upstream dependency |

## Known issues

1. **Full chain never tested** — see frenken-lab/graphids#25. No end-to-end KD run has
   been recorded.
2. **DGI has no KD support** — `DGIModule` has no `distillation_config` wiring in
   `inject_loss_fn` (only `"gat"` and `"vgae"` are in `_LOSS_MODEL_TYPES`,
   `build.py:13`).
3. **Teacher stored via `__dict__`** — bypasses `nn.Module` registration so Lightning
   never auto-transfers to GPU. `_run_teacher_on` handles per-step movement. Localized
   to `_attach_teacher()` in `distillation.py:38`.
