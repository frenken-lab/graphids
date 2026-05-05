# Config Architecture

> Python composition (`graphids/plan/`) → Pydantic validation
> (`BlueprintArray`) → direct instantiation (`orchestrate._instantiate`).
> For composer rules, env vars, path scheme, and observability wiring,
> see `.claude/rules/config-system.md`.

The pre-2026-05-04 jsonnet layer (`gojsonnet` + `configs/*.libsonnet`)
was deleted. Don't reach for it. New plans are Python.

---

## 1. CLI surface

```
python -m graphids run <plan> --dataset X --seed N [-o plan.json]
  -> __main__.py
  -> cli.commands:run_cli
  -> importlib.import_module(f"graphids.plan.plans.{plan}")
  -> mod.build(dataset=X, seed=N)              # returns list[dict]
  -> BlueprintArray.model_validate(rendered)   # raises on schema drift
  -> sys.stdout / output file (JSON array)
```

`<plan>` is a dotted module name under `graphids.plan.plans`
(e.g. `unsupervised`, `ofat`, `smoke.gat_taunorm`).

```
python -m graphids exec --row '<row JSON>' [--ckpt-path X]
  -> orchestrate.run_row(row)                  # in-process dispatch
```

```
python -m graphids submit --row '<row JSON>' --cluster <c> [--length L]
  -> graphids.slurm.submit.submit_row(row, ...)
  -> Parsl SlurmProvider.submit (sbatch carries the literal
     `python -m graphids exec --row '...'` command — no pickle)
  -> prints jid on stdout
```

The four-step chassis (render → blueprint → exec → submit) is codified
in `.claude/rules/single-submission-primitive.md`.

---

## 2. Composition layers

```
graphids/plan/
├── lib.py                # class-path string constants + spec(...) helper
│                         # + composing primitives (can_bus, graph_dm,
│                         #   fusion_dm, curriculum) + REWARD constant
├── compose.py            # compose(...) + fusion(...) — single composer
│                         # plus archetype wrapper. Owns trainer_base /
│                         # callbacks_base assembly.
├── blueprint.py          # Pydantic schemas: ClassPath, TrainerCfg,
│                         # RenderedConfig, TrainRow/CmdRow/ExtractRow/
│                         # AnalyzeRow, BlueprintArray
├── row.py                # RowSpec dataclass + .fit() / .test() / extract()
└── plans/                # build(dataset, seed) -> list[dict]
    ├── unsupervised.py / fusion.py / ofat.py
    └── ops/              # one-shot ops plans (analyze / extract / smoke)
```

Path math lives in `graphids/plan/catalog.py` (separate `config`
package — historical name, no shim). Composer + plans + lib import
from there directly.

### Lib (class-path strings + `spec()`)

`spec(cls_path, **init_args)` is a 3-line dict builder. The 11 trivial
"primitive functions" (`gat`, `focal`, `ce`, `bandit`, …) were collapsed
into named string constants here; defaults live with the model class
(e.g. `GAT.__init__` reads its own `_SCALES` table for `scale="small"`
vs `"large"`). Only four primitives stay as functions because they do
real work: `can_bus` (registry validation), `graph_dm` (conditional
optional knobs), `fusion_dm` (`states_dir(...)` derivation), and
`curriculum` (deepcopy + `reduction='none'` injection on the base loss).

```python
from graphids.plan.lib import GAT, FOCAL, spec, can_bus, graph_dm

spec(GAT, scale="large", dropout=0.3)
graph_dm(source=can_bus(dataset="hcrl_sa", seed=42))
```

### Composer

`compose(model, data, *, loss=None, meta, monitor, mode, run_mode,
trainer_overrides, upstreams, ...)` returns a frozen `RowSpec` whose
`rendered` is a typed `RenderedConfig` (Pydantic, frozen,
`extra="forbid"`). The composer is the single site that:

1. Computes `run_dir` via `graphids.plan.catalog.run_dir(...)`.
2. Merges the optional `loss` block into `model.init_args.loss_fn`
   via an explicit `update` call — no deep-merge magic.
3. Builds the universal callbacks trio (checkpoint, early_stopping,
   mlflow) and the `trainer.callbacks` list (alphabetical key order
   for byte-identical re-renders).
4. Constructs `RenderedConfig(model=ClassPath(...), data=ClassPath(...),
   trainer=TrainerCfg(...), callbacks={...}, seed_everything=...)` —
   typo'd field at compose time raises `pydantic.ValidationError`.

`fusion(...)` is a thin wrapper that auto-derives the `[vgae, focal]`
upstreams from `meta` and applies the fusion-fixed trainer overlay
(`precision="32-true"`, `gradient_clip_val=None`, `max_epochs=1500`,
`log_every_n_steps=10`).

### Plans

Each plan exposes `def build(*, dataset: str, seed: int) -> list[dict]`.
Plans compose `spec(...)` blocks + composers, then call
`RowSpec.fit(name)` / `.test(name)` to emit row dicts. The `extract()`
top-level function in `row.py` builds one-shot fusion-feature rows.

OFAT (`plans/ofat.py`) is the largest surface — uses a declarative
`SWEEPS` dict (axis → variant → kwargs) over a single `gat_row(...)`
helper. Adding an ablation row = adding a dict entry. The dict closes
over `dataset` / `seed` / `vgae_ckpt` since variants reference them.

---

## 3. Pydantic validation

`graphids.plan.blueprint.BlueprintArray` is a `RootModel[list[Row]]`
where `Row` is a discriminated union by `action`:

| Action | Class | Notes |
|---|---|---|
| fit / test | `TrainRow` | `rendered_config: RenderedConfig` (typed end-to-end), `meta`, `identity`, `upstreams`, `resources` |
| cmd | `CmdRow` | arbitrary shell command |
| extract | `ExtractRow` | one-shot fusion-feature extraction (idempotent on `output_dir`) |
| analyze | `AnalyzeRow` | per-checkpoint artifacts (CKA / embeddings / landscape / fusion-policy) |

All rows: `extra="forbid"`, `frozen=True`. `AnalyzeRow` runs an
`@model_validator` that enforces conditional dependencies
(`cka=True ⇒ cka_teacher_ckpt`, `fusion_policy=True ⇒ vgae_ckpt_path`,
etc.).

`RenderedConfig` is itself typed (`model: ClassPath`, `data: ClassPath`,
`trainer: TrainerCfg`, `callbacks: dict[str, ClassPath]`,
`seed_everything: int`). `init_args` stay free-form (`dict[str, Any]`)
since per-class kwargs aren't worth enumerating in a schema.

---

## 4. Direct instantiation

`graphids.orchestrate._instantiate(spec)` walks `{class_path, init_args}`
blocks recursively:

```python
def _instantiate(spec):
    rec = lambda v: _instantiate(v) if isinstance(v, dict) and "class_path" in v else v
    init = {k: rec(v) for k, v in spec.get("init_args", {}).items()}
    mod, _, attr = spec["class_path"].rpartition(".")
    return getattr(importlib.import_module(mod), attr)(**init)
```

This is the same recursive-class_path pattern as
`hydra.utils.instantiate(cfg, _recursive_=True)`; graphids reuses the
shape with `class_path` instead of `_target_`.

---

## 5. Path scheme

Path math lives in `graphids/plan/catalog.py`. Plans + composer call
`run_dir(dataset, group, variant, seed)` and `best_ckpt(...)` directly:

```
{$GRAPHIDS_RUN_ROOT}/{dataset}/ablations/{group}/{variant}/seed_{N}
```

`GRAPHIDS_RUN_ROOT` is required (no default — fail-fast in
`_run_root()`). `GRAPHIDS_LAKE_ROOT` is the cross-user shared root
(MLflow DB, raw CSVs, preprocessed cache). See
`.claude/rules/data-layout.md`.

---

## 6. Running

```bash
# Render a Python plan to a row array.
python -m graphids run ofat --dataset hcrl_sa --seed 42 -o plan.json

# Login-node smoke (in-process; non-SLURM).
jq -c '.[0]' plan.json | xargs -I{} python -m graphids exec --row {}

# SLURM submission.
jq -c '.[]' plan.json | while read row; do
    python -m graphids submit --row "$row" --cluster pitzer --length long
done
```

---

## 7. Adding a new plan

1. Create `graphids/plan/plans/<name>.py` with
   `def build(*, dataset: str, seed: int) -> list[dict]`.
2. Compose with `spec(...)` + composing primitives + `compose(...)`.
3. Add a smoke entry in `tests/configs/test_plans_smoke.py::PLANS`.
4. Done — `graphids run <name>` works immediately.

If the plan needs new declarative state, add a field to the relevant
row class in `graphids/plan/blueprint.py` (`extra="forbid"` will
reject unknown keys until you do).
