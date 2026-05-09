# `graphids.plan` — Python plans → typed rows → JSON

Plans are Python modules that emit a validated JSON description of one
SLURM-shaped experiment batch. The JSON is the boundary between authoring
(login node, pure) and execution (compute node, re-imports current source).
Re-rendering is only required when a source change alters either what the
composer writes into the JSON or which kwargs a `class_path` will accept.

## Package layout

| File | Role |
|---|---|
| [`schema.py`](schema.py) | Pydantic models — `Plan`, `TrainRow`, `CacheRow`, `ExtractRow`, `AnalyzeRow`, `RenderedConfig`, `ClassPath`, `TrainerCfg`. All `frozen=True, extra="forbid"`. |
| [`compose.py`](compose.py) | `compose()` / `fusion()` → frozen `RowSpec`; `RowSpec.fit(name)` / `.test(name)` emit `TrainRow`-shaped dicts. `extract()` is the one-shot extraction-row builder. |
| [`config.py`](config.py) | Typed discriminated-union configs: `GATCfg`, `VGAECfg`, `FocalLossCfg`, `GraphDMCfg`, … and their union aliases `ModelCfg`, `LossFn`, `DataCfg`, `DifficultyCfg`. All `frozen=True, extra="forbid"`. |
| [`primitives.py`](primitives.py) | Typed factory functions: `gat()`, `vgae()`, `focal()`, `graph_dm()`, `fusion_dm()`, `curriculum()`, … Return frozen Pydantic configs. Only `can_bus()` does validation (catalog membership). |
| [`render.py`](render.py) | `render_plan(plan_module, dataset=, seed=, filter_glob=)` — imports the plan module, calls `build()`, threads `plan_id` + `git_sha` + `plan_module` onto fit/test rows, validates as `Plan`. Single call site for `gx run` and `gx plans describe`. |
| [`identity.py`](identity.py) | `mint_plan_id()` (UUIDv7, lex-sortable) + `git_sha()` (working-tree short SHA). |
| [`plans/`](plans/) | User-authored `build(*, dataset, seed) -> list[dict]` modules. Existing axes: `ablations.{fusion,supervised,unsupervised}`, `data.rebuild_cache`, `smoke.gat_taunorm`. Add a new file here, name it after its axis. |

## Render → submit → exec flow

- **render** — `gx run <plan_module> -d <dataset> -s <seed> -o plan.json`. Pure, login-node, JSON only. No SLURM, no MLflow, no network.
- **submit** — `gx plans submit --plan plan.json -C pitzer`. Sbatch body is the literal `python -m graphids exec --row '<json>' [--ckpt-path X]`. Row JSON is frozen here ([`chassis-invariants.md`](../../.claude/rules/chassis-invariants.md) §2).
- **exec** — compute node re-imports current source, `graphids.orchestrate.run_row` match-dispatches on the typed config tag (`cfg.type`) to instantiate model / loss / datamodule; callbacks still use `{class_path, init_args}` blocks via `_instantiate_callback`.

## Re-render decision table

The boundary: **does the change alter what `compose()` writes into the JSON, or change what the typed config fields mean at exec time?** YES → re-render. Otherwise the queued sbatch picks it up automatically on exec.

| Source change in… | Re-render? | Why |
|---|---|---|
| `graphids/plan/{schema,compose,config,primitives,render,identity}.py` | **YES** | Composition output structure or identity tagging changes. |
| `graphids/plan/plans/<your_plan>.py::build()` body | **YES** | The plan IS the producer of the JSON. |
| `build_model` / `build_loss` / `build_datamodule` in `orchestrate.py` adds a new **required** constructor kwarg | **YES** | Frozen config field won't satisfy the new contract. |
| Constructor adds a new **optional** kwarg (with default) | NO | Re-import on exec picks up the new default; config field is absent → default applies. |
| Constructor renames an existing kwarg | **YES** | Frozen config field name is in the row JSON; new name won't match. |
| Callback `class_path` `__init__` adds a required kwarg | **YES** | `init_args` in the frozen JSON won't satisfy it. |
| Model `forward()` / loss math / `training_step` internals | NO | Sbatch re-imports source on exec ([`chassis-invariants.md`](../../.claude/rules/chassis-invariants.md) §2). |
| New file `core/models/_<helper>.py`, consumed only by an existing class internally | NO | No `class_path` reference in JSON ⇒ invisible to render. |
| `graphids/cli/plans/*.py` (consumer of JSON) | NO | Consumer, not producer. |
| `graphids/orchestrate.py` (run_row dispatch / SIGUSR2 wiring) | NO | Re-imported on exec. |
| `configs/resources/submit_profiles.json` (sbatch profile) | NO | Read at submit-time, not from row JSON. |
| `graphids/_mlflow.py` (callback / identity_tags) | NO | Consumed at exec time; tags emitted live. **Exception**: removing/renaming a key in `_TAG_KEYS` violates invariant §4 — fix migration tooling, then re-render is moot (tags are emitted, not stored in row JSON). |

## Re-check before resubmit

Each row carries the render-time SHA in `row.git_sha`. To audit:

```bash
RENDER_SHA=$(jq -r '.rows[0].git_sha' plan.json)
git diff $RENDER_SHA..HEAD -- graphids/plan/                      # any output → re-render
git diff $RENDER_SHA..HEAD -- 'graphids/core/**/*.py' \
  | grep -E '^\+.*def __init__|^\+.*: [A-Z][a-zA-Z_]+,?$'         # added required kwarg → re-render
```

If both diffs are empty, the queued JSON still describes the same composition; source-level changes (model internals, loss math) flow to exec automatically.

## Cross-references

- [`.claude/rules/chassis-invariants.md`](../../.claude/rules/chassis-invariants.md) — invariants §1 (render purity), §2 (drift resistance), §4 (reproduction contract via `plan_id` + `plan_module` + `plan_args` + `git_sha` + `row_name`).
- [`.claude/rules/config-system.md`](../../.claude/rules/config-system.md) — composition rules, null preservation, observability wiring.
- [`docs/reference/config-architecture.md`](../../docs/reference/config-architecture.md) — full architecture doc.
