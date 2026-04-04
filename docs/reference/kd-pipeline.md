# Knowledge Distillation Pipeline

> Status: Wired but untested | See frenken-lab/graphids#25 for gaps

## What KD does

Large ("teacher") models are trained first, then their knowledge is compressed into
small ("student") models via auxiliary KD loss terms. The student trains on both the
task loss and a distillation loss that aligns its representations with the teacher's.

- **GAT KD**: Hinton soft-label — `KL(student/T || teacher/T) * T²`, blended as
  `alpha * kd_loss + (1 - alpha) * task_loss`
- **VGAE KD**: Latent-space alignment — weighted sum of latent and reconstruction
  losses between student and teacher embeddings

## How to enable KD in a recipe

Add a `kd:` block to any sweep entry. For orchestrated runs, `teacher_config`
is **required** — it names the recipe config that produces the teacher
checkpoint. Silent scale-based inference was removed (see
`docs/reference/orchestration-risks.md` item #2 — the old behavior rewired the
student to a different teacher when recipe keys were reordered).

```yaml
sweeps:
  - model_family: gat
    stage: curriculum
    scale: small
    kd:
      alpha: 0.7                       # blend weight (0 = task only, 1 = KD only)
      teacher_config: gat_curriculum_large  # REQUIRED: recipe config key of the teacher
      teacher_scale: large              # used by the dev path (prepare_kd) only
      temperature: 4.0                  # softmax temperature (GAT only)
```

Planning validates at enumeration time that `teacher_config`:

- names a config that exists in the recipe (else: "does not name a config"),
- has no KD auxiliaries of its own (else: "has its own auxiliaries — teachers
  must train without KD"),
- produces an asset for the student's current stage (else: "does not produce a
  '<stage>' asset").

The named config's asset is wired as an explicit upstream dependency so dagster
schedules the teacher first.

## Data flow

```
Recipe kd: block
  │
  ▼
KDEntry validation (contracts.py:12)
  8 fields: type, alpha, teacher_config, teacher_scale, temperature,
  model_path, vgae_latent_weight, vgae_recon_weight
  │
  ▼
_expand_sweep emits auxiliaries list (recipe_expand.py:117-119)
  over["auxiliaries"] = [kd_payload]
  │
  ▼
enumerate_assets (planning.py)
  - Sets kd_tag="_kd" on StageConfig (distinct identity hash)
  - Appends config/models/{family}/kd.yaml to config chain
  - _resolve_kd_teachers: for each KD aux, looks up teacher_config by name,
    validates it exists/has no aux/produces the student's stage, wires as
    upstream. Fails loud on any mismatch.
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
   TypedDict validation. See frenken-lab/graphids#19 (P2.4).
2. **Full chain never tested** — see frenken-lab/graphids#25.
3. **Teacher stored via `__dict__`** — bypasses `nn.Module` registration, so
   Lightning never auto-transfers to GPU. `teacher_on_device` context manager
   handles per-step movement. Deliberate but fragile.
