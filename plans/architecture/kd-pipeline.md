# Knowledge Distillation Pipeline

> Status: Wired but untested | See `issues/kd-pipeline-untested.md` for gaps

## What KD does

Large ("teacher") models are trained first, then their knowledge is compressed into
small ("student") models via auxiliary KD loss terms. The student trains on both the
task loss and a distillation loss that aligns its representations with the teacher's.

- **GAT KD**: Hinton soft-label — `KL(student/T || teacher/T) * T²`, blended as
  `alpha * kd_loss + (1 - alpha) * task_loss`
- **VGAE KD**: Latent-space alignment — weighted sum of latent and reconstruction
  losses between student and teacher embeddings

## How to enable KD in a recipe

Add a `kd:` block to any sweep entry:

```yaml
sweeps:
  - model_family: gat
    stage: curriculum
    scale: small
    kd:
      alpha: 0.7            # blend weight (0 = task only, 1 = KD only)
      teacher_scale: large   # teacher checkpoint scale to resolve
      temperature: 4.0       # softmax temperature (GAT only)
```

The recipe must also include a matching large-scale entry (or the large teacher
checkpoint must already exist on disk):

```yaml
  - model_family: gat
    stage: curriculum
    scale: large             # teacher — trains first, student depends on it
```

## Data flow

```
Recipe kd: block
  │
  ▼
_KDSpec validation (recipe_expand.py:11-21)
  7 fields: type, alpha, teacher_scale, temperature,
  model_path, vgae_latent_weight, vgae_recon_weight
  │
  ▼
_expand_sweep emits auxiliaries list (recipe_expand.py:117-119)
  over["auxiliaries"] = [kd_payload]
  │
  ▼
enumerate_assets (planning.py:84-151)
  - Sets kd_tag="_kd" on StageConfig (distinct identity hash)
  - Appends config/models/{family}/kd.yaml to config chain
  - Scans recipe for large-scale teacher → upstream dagster dep
  - Stores raw payload in StageConfig.kd_overrides
  │
  ▼
ConfigResolver.resolve (resolve.py:101-105)
  JSON-encodes: runtime_overrides["model.init_args.auxiliaries"] = '[{...}]'
  │
  ▼
TrainingContract.to_override_dict (ops.py:143-145)
  Passes runtime_overrides through to CLI arg dict
  │
  ▼
LightningCLI (via run_lightning)
  Parses --model.init_args.auxiliaries=[{"type":"kd",...}]
  jsonargparse validates against list[KDAuxiliary] | None
  │
  ▼
Model.__init__(auxiliaries=[...])
  save_hyperparameters() stores auxiliaries
  │
  ▼
Model._build() → prepare_kd (_training.py:195-240)
  1. Finds auxiliary with type=="kd"
  2. Resolves teacher path: kd.model_path or checkpoint_path()
  3. load_inner_model() → safe_load_checkpoint
  4. teacher.requires_grad_(False)
  5. VGAE: creates nn.Linear projection if latent_dim differs
  │
  ▼
Model.training_step
  GAT: alpha * kd_loss + (1-alpha) * task_loss
  VGAE: task_loss + latent_weight * latent_kd + recon_weight * recon_kd
```

## Config files

| File | Role |
|------|------|
| `config/models/vgae/kd.yaml` | VGAE KD overlay defaults (alpha, latent/recon weights) |
| `config/models/gat/kd.yaml` | GAT KD overlay defaults (alpha, temperature) |

These overlays are appended to the config chain when `include_kd_overlay=True`.
They provide defaults that the resolver's `runtime_overrides` injection overwrites
at CLI parse time.

## Key code

| Symbol | Location | Role |
|--------|----------|------|
| `_KDSpec` | `config/recipe_expand.py:11` | Recipe-side KD config schema |
| `KDAuxiliary` | `core/models/_training.py:92` | Model-side KD config TypedDict |
| `prepare_kd` | `core/models/_training.py:195` | Teacher loading + projection |
| `checkpoint_path` | `config/paths.py:149` | Teacher checkpoint path resolution |
| `kd_overrides` | `orchestrate/planning.py:38` | StageConfig field for KD payload |
| `kd.yaml` overlays | `config/models/{vgae,gat}/` | Default KD hyperparameters |

## Known issues

1. **`_KDSpec` (7 fields) vs `KDAuxiliary` (3 fields)** — 4 extra fields bypass
   TypedDict validation. See `issues/config-system-overhaul.md` P2.4.
2. **Full chain never tested** — see `issues/kd-pipeline-untested.md`.
3. **Teacher stored via `__dict__`** — bypasses `nn.Module` registration, so
   Lightning never auto-transfers to GPU. `teacher_on_device` context manager
   handles per-step movement. Deliberate but fragile.
