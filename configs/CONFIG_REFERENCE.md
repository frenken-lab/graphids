# Config Reference

> Audited: 2026-05-01 (post-mirror redesign).
> Architecture: see `~/plans/graphids-jsonnet-design.md`.

The `configs/` tree mirrors `graphids/core/` so each leaf is a primitive
configuration of one Python class. There is no spec layer — plans bind
primitives directly via the barrel `configs/index.libsonnet`.

```
configs/
├── index.libsonnet       BARREL — single import: `local g = import '../index.libsonnet'`
├── models/               ↔ graphids/core/models/
│   ├── autoencoder/{vgae,dgi}.libsonnet
│   ├── supervised/gat.libsonnet
│   └── fusion/{bandit,dqn,mlp,weighted_avg,_reward}.libsonnet
├── losses/               ↔ graphids/core/losses/
│   └── {focal,ce,weighted_ce}.libsonnet
├── data/                 ↔ graphids/core/data/
│   ├── datasets.json     dataset registry
│   ├── source/can_bus.libsonnet
│   └── datamodule/{graph,fusion}.libsonnet
├── compose/              ARCHETYPE COMPOSERS — no Python equivalent (pure config glue)
│   └── {unsupervised,supervised,fusion}.libsonnet
├── _kit/                 INFRASTRUCTURE — trainer, callbacks, validate, row builder
│   └── {trainer,callbacks,validate,row}.libsonnet
├── plans/                ENTRY POINTS — what Python evaluates
│   └── {unsupervised,supervised,supervised_ablations,fusion,ofat}.jsonnet
└── resources/submit_profiles.json
```

**Three extensions, three roles**: `.libsonnet` = function/value (imported),
`.jsonnet` = top-level entry point (evaluated), `.json` = pure data.

---

## 1. Datasets

Registry: `data/datasets.json`. One entry per dataset with metadata that
doesn't reduce to the dataset name (attack types, notes).

| Dataset | Source | Attack types |
|---|---|---|
| `hcrl_ch` | HCRL Challenge | dos, fuzzing, gear_spoofing, rpm_spoofing |
| `hcrl_sa` | HCRL Scenario Anomaly | mixed |
| `set_01`–`set_04` | Automotive CAN | mixed, suppress, masquerade |

Source primitive (`data/source/can_bus.libsonnet`) does the registry
lookup + emits the `CANBusSource` block. Unknown dataset names fail loudly
at render with a list of valid options.

Datamodules (`data/datamodule/`):

| File | Class | Used by |
|---|---|---|
| `graph.libsonnet` | `GraphDataModule` | unsupervised + supervised archetypes |
| `fusion.libsonnet` | `FusionDataModule` | fusion archetype (cache_dir IS the source) |

---

## 2. Models

Each architecture has its own libsonnet under `models/<family>/`:

| Family | Libsonnet | Module |
|---|---|---|
| Unsupervised | `models/autoencoder/{vgae,dgi}.libsonnet` | `core/models/autoencoder/{vgae,dgi}_module.py` |
| Supervised | `models/supervised/gat.libsonnet` | `core/models/supervised/gat_module.py` |
| Fusion | `models/fusion/{bandit,dqn,mlp,weighted_avg}.libsonnet` | `core/models/fusion/*.py` |

Each is `function(scale='small', conv_type='gatv2', ...) → {model: {...}}`.

### Scale axis
Per-architecture `_scales = {small, large}` maps embedded in each model
libsonnet. Not a primitive — it's a TLA parameter the model primitive consumes.

### Fusion methodology
Reward shaping constants live in `models/fusion/_reward.libsonnet`,
imported by `bandit.libsonnet` + `dqn.libsonnet` (the two RL methods).
MLP + weighted_avg are supervised baselines without reward signals.

### Loss primitives
| File | Type fragment (consumed by Python `inject_loss_fn`) |
|---|---|
| `losses/focal.libsonnet` | `{type: 'focal', gamma: 2.0}` |
| `losses/ce.libsonnet` | `{type: 'ce'}` |
| `losses/weighted_ce.libsonnet` | `{type: 'weighted_ce', weights: [...]}` |

---

## 3. Archetype composers (`compose/`)

Three composers own group-shared composition:

| Composer | Used by | Owns |
|---|---|---|
| `unsupervised.libsonnet` | VGAE, DGI | label_filter='benign' default, monitor flexible |
| `supervised.libsonnet` | GAT (all loss/sampling/scaler/id_encoding bindings) | full train set, monitor=val_auroc, patience=50 |
| `fusion.libsonnet` | bandit, dqn, mlp, weighted_avg | cpu mode, precision=32-true, max_epochs=1500, patience=200, upstreams=[vgae, focal] |

Each composer enforces:
- `default_root_dir` from `paths.run_dir(meta...)` — single source
- `seed_everything = meta.seed`
- `v.spec(...)` apex validation (every binding contract-checked)
- Trainer base + archetype-specific overrides + per-call `trainer_overrides`

Adding a new archetype = one new file in `compose/`.

---

## 4. Callbacks (`_kit/callbacks.libsonnet`)

Universal trio + extras knob:

```jsonnet
function(monitor, mode, patience, extras={}) → {
  callbacks: { checkpoint, early_stopping, mlflow } + extras
}
```

- **Universal** (mandatory): checkpoint, early_stopping, mlflow.
- **Optional** (per-binding): pass via composer's `callback_extras={...}`
  (used by curriculum bindings to add `CurriculumEpochCallback`).
- **Auto-injected** (Python-side): `VRAMDriftCallback` added at instantiation
  if CUDA available. Not in jsonnet.

`trainer.callbacks` (the LIST Lightning consumes) is late-bound from
`$.callbacks` (the DICT) at the apex via:
```jsonnet
callbacks: [$.callbacks[k] for k in std.objectFields($.callbacks)]
```
in `_kit/trainer.libsonnet`. Any callback added to the dict (universal trio,
extras) is auto-listed.

---

## 5. Plans

A plan is `function(dataset, seed) → list[row]`. Each row is self-contained
(rendered_config + identity + upstreams + resources). Plans emit a JSON
array, not JSONL.

Plans bind primitives directly — there is no spec layer:

```jsonnet
local g = import '../index.libsonnet';
function(dataset, seed)
  local vgae = g.compose.unsupervised(
    model = g.models.autoencoder.vgae(),
    data  = g.data.datamodule.graph(
      source       = g.data.source.can_bus(dataset, seed),
      label_filter = 'benign',
    ),
    monitor = 'val_discrimination_ratio',
    meta    = { group: 'unsupervised', variant: 'vgae', ...},
  );
  [g.row.fit('vgae', vgae), g.row.test('vgae', vgae)]
```

| Plan | Rows | Purpose |
|---|---|---|
| `unsupervised.jsonnet` | 4 | VGAE + DGI fit/test |
| `supervised.jsonnet` | 2 | GAT focal fit/test (single-binding spike) |
| `fusion.jsonnet` | 8 | All 4 fusion methods fit/test |
| `supervised_ablations.jsonnet` | 3 | loss + upstreams stress test |
| `ofat.jsonnet` | 22 | Full one-factor-at-a-time sweep across every axis |

Row builder (`_kit/row.libsonnet`):
- `row.fit(name, rendered)` — fit row
- `row.test(name, rendered)` — test row (action='test', same identity)
- `row.cmd(name, command, mode, length)` — non-binding command row

Identity strings synthesized via `std.format` from `_meta`.

---

## 6. Resources

Two strings on each row's `resources`: `mode` (gpu/cpu) + `length` (short/long).
Cluster-specific numerics (`partition`, `cores_per_node`, `mem_per_node`,
`walltime`, `gpus_per_node`) come from `resources/submit_profiles.json`
keyed `[mode][cluster][length]` — keys map 1:1 to `parsl.providers.SlurmProvider`.

The blueprint is **portable across clusters** — never pre-bake cluster numbers.

---

## 7. Native callbacks (Python ↔ jsonnet bridge)

```
ext_codes:        run_root, overrides
tla_codes:        per-call top-level args (dataset, seed, scale, ...)
native_callbacks: paths.run_dir, paths.best_ckpt, paths.states_dir
```

Three natives — all FS-rooted paths, source-of-truth `graphids/config/paths.py`.
Identity strings (run_name, jobname) are computed in jsonnet via `std.format`,
not via natives.

Binding: `gojsonnet` (Go implementation, full stdlib including SHA family).

---

## 8. Removed (post-mirror redesign 2026-05-01)

| Old | New |
|---|---|
| `configs/specs/<group>/<v>.libsonnet` (16 files) | inlined in `configs/plans/*.jsonnet` |
| `configs/_lib/{trainer,callbacks,validate,row}.libsonnet` | `configs/_kit/...` |
| `configs/_lib/compose/*.libsonnet` | `configs/compose/...` |
| `configs/_lib/source/*.libsonnet` | `configs/data/source/...` |
| `configs/_lib/datamodule/*.libsonnet` | `configs/data/datamodule/...` |
| `configs/_lib/loss/*.libsonnet` | `configs/losses/...` |
| `configs/_lib/models/*.libsonnet` | `configs/models/<family>/...` |
