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
Jsonnet recipe expansion emits auxiliaries list (configs/_lib/recipes.libsonnet)
  auxiliaries: [kd_payload]
  │
  ▼
enumerate_assets (planning.py)
  - Sets kd_tag="_kd" on StageConfig (distinct identity hash)
  - _resolve_kd_teachers: for each KD aux, looks up teacher_config by name,
    validates it exists/has no aux/produces the student's stage, wires as
    upstream. Fails loud on any mismatch.
  - Stores raw payload in StageConfig.kd_overrides
  │
  ▼
ConfigResolver.resolve (resolve.py)
  Packs kd_overrides into jsonnet_tla["auxiliaries"] via
  graphids.orchestrate.contracts.build_tla_dict (typed dict, not stringified)
  │
  ▼
render_config(jsonnet_path, jsonnet_tla)  (config/jsonnet.py)
  Stage libsonnet merges vgae.kd / gat.kd overlay when auxiliaries is
  non-empty, producing model.init_args.auxiliaries = [{type: "kd", ...}]
  │
  ▼
validate_config(rendered)  (config/schemas.py)
  Pydantic gate: rejects null list fields, enforces class_path namespacing
  │
  ▼
graphids.instantiate.instantiate (instantiate.py)
  _coerce_kd_auxiliaries: each list item promoted dict → SimpleNamespace
  so Model._install_kd_teacher can call getattr(a, "type", None)
  Model(**init_args) — direct importlib instantiation, no jsonargparse
  │
  ▼
Model.__init__(auxiliaries=[SimpleNamespace(type="kd", alpha=..., ...)])
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
| `KDEntry` | `contracts.py` | Recipe-side Pydantic schema (superset) |
| `KDAuxiliary` | `core/models/_training.py` | Student-side TypedDict (runtime subset) |
| `_install_kd_teacher` | `core/models/_training.py` | `GraphModuleBase` method: resolves KD cfg, loads + freezes teacher, stores it off Lightning's auto-transfer path |
| `_kd_loss` | `vgae.py`, `gat.py` | Per-model KD loss shape (VGAE dual-signal MSE, GAT Hinton soft-label KL) |
| `_apply_kd` | `core/models/_training.py` | Convex combo: α·kd + (1−α)·task |
| `prepare_kd` | `core/models/_training.py` | Teacher checkpoint path resolution + load + projection layer |
| `checkpoint_path` | `config/paths.py` | Teacher checkpoint path from identity keys |
| `kd_overrides` | `orchestrate/planning/shared.py` | StageConfig field for KD payload |
| `kd.yaml` overlays | `config/models/{vgae,gat}/` | Default KD hyperparameters |

## Schema relationship

`KDEntry` and `KDAuxiliary` are deliberately not identical. `KDEntry` is the
recipe-side superset (8 fields, Pydantic `extra="forbid"`). `KDAuxiliary` is the
runtime subset that LightningModules actually receive (7 fields, TypedDict for
jsonargparse validation). The difference is **`teacher_config`**, which is a
planning-only field: the orchestrator consumes it to wire an upstream
dependency and resolves it into `model_path` before the student module ever
sees the config. Students never receive `teacher_config` directly.

Both docstrings cross-reference each other so the subset relationship is
discoverable from either side.

## Known issues

1. **Full chain never tested** — see frenken-lab/graphids#25. The
   `teacher_on_device` decorator bug (fixed 2026-04-04, session 18) would have
   crashed any KD training on the first step with `TypeError: 'generator' object
   does not support the context manager protocol`. This alone explains why
   no KD run has ever been cited end-to-end. The code path is now valid but
   still unexercised.
2. **DGI has no KD support** — `DGIModule` does not call `_install_kd_teacher()`
   and has no `_kd_loss()`. If claim 6 ablation extends to DGI-large → DGI-small,
   add these two hooks.
3. **Teacher stored via `__dict__`** — bypasses `nn.Module` registration, so
   Lightning never auto-transfers to GPU. `teacher_on_device` context manager
   handles per-step movement. The hack is localized to
   `GraphModuleBase._install_kd_teacher()` (session 18) — one place to worry
   about.
