# Orchestration Risks — Known Fragile & Complex Areas

> Status: **living audit** | Last reviewed: 2026-04-04
>
> Companion to [`orchestration.md`](orchestration.md) and [`3-chain.md`](3-chain.md).
> This doc inventories the load-bearing complexity that remains in `graphids/orchestrate/`
> after the duplicated-helpers and dagster-native-checks cleanups landed in commit `63f59e8`.
> Items here are either inherently hard (domain semantics) or have been deferred because
> the fix is non-trivial.
>
> **Resolved since last review:**
> - Tier 1 item #3 (validation zoo in `resolve.py`) — refactored into a
>   `ValidationRule` dataclass + `_RULES` tuple.
> - Tier 1 item #2 (KD teacher resolution in `planning.py`) — replaced with an
>   explicit `teacher_config` field that names the teacher recipe config by key.
>   See each item body for details.

## Tier 1 — Real fragility

### 1. Identity-key resolution in `planning.py`

**Location:** `planning.py:15–27`, `_identity_value()` + `_RECIPE_TO_IDENTITY`.

The identity keys declared in `pipeline.yaml` don't match the field names in the recipe
schema, so a translation layer with four stacked special cases exists:

```python
_RECIPE_TO_IDENTITY = {"fusion_method": "method"}   # one-entry dict

def _identity_value(key, merged, stages):
    if key == "gat_stage":                          # (1) derive from stages list
        return "curriculum" if "curriculum" in stages else "normal"
    _get = merged.get if isinstance(merged, dict) else lambda k, d=None: getattr(merged, k, d)
    for rk, ik in _RECIPE_TO_IDENTITY.items():      # (2) rename fusion_method → method
        if ik == key:
            return _get(rk)
    val = _get(key)
    if key == "model_type" and val is None:         # (3) default model_type → "vgae"
        return "vgae"
    return val
```

**Why it's fragile:**

- Each branch exists because of a specific recipe-vs-topology naming clash. Comments
  explain *what* the code does, not *why* the clash exists.
- A fifth clash gets added as another branch with no framework pushback.
- No test asserts that every identity key in every stage resolves to a non-`None` value
for every recipe in `configs/recipes/`. `compute_identity_hash` (`config/paths.py:158`)
  raises `KeyError` on missing keys, but the error lands far from the cause.

**What would fix it:** either unify naming in `pipeline.yaml` and the recipe schema
(breaking change for every recipe), or make the translation table explicit and exhaustive
and raise on unknown keys instead of falling through.

### 2. KD teacher resolution in `planning.py` — **RESOLVED**

**Status:** Replaced with explicit `teacher_config` naming. `KDEntry` gained a
`teacher_config: str | None` field (`contracts.py:12`); `planning.py` gained a
`_resolve_kd_teachers` helper that iterates **all** KD auxiliaries (fixing the
old `[0]`-only bug as a side effect) and looks up each teacher by name. Four
validations run on every student: teacher_config is set, names an existing
config, that config has no auxiliaries of its own, and that config produces
an asset for the student's current stage. Each mismatch raises with the
student name + stage + set of valid alternatives.

The old scale-search + first-match + hardcoded `("autoencoder", "curriculum",
"normal")` fallback loop is deleted. Multi-teacher KD is now wire-correct:
each aux gets resolved independently, no silent first-match bias.

**Tests:** `tests/orchestrate/test_overrides.py::TestKDTeacherResolution`
covers the happy path, all four error paths, and an explicit order-invariance
test (two recipes with the same configs in opposite key order must produce
identical upstream wiring).

**Migration cost paid:** zero active recipes used KD at refactor time, so
no recipe YAML changes were needed. Future KD recipes must set
`teacher_config` explicitly — the old `teacher_scale`-based inference would
have raised at runtime anyway.

**Out of scope (still fragile):** the dev-path runtime (`prepare_kd` in
`core/models/_training.py:259–270`) still recomputes the teacher checkpoint
path from `teacher_scale` via `checkpoint_path()`. It works because dev users
aren't affected by recipe-key ordering, but it duplicates the teacher-identity
assumption. A follow-up could plumb the resolved teacher ckpt through
`upstream_ckpt_paths` so `prepare_kd` just reads the path instead of
recomputing.

**Location (historical):** `planning.py:152–164`.

Original description — kept for context on why the refactor landed:

KD student configs named their teacher only by `teacher_scale`. Planning resolved the
actual teacher by walking all other recipe configs looking for a match:

```python
if has_kd:
    teacher_scale = merged.auxiliaries[0].teacher_scale if merged.auxiliaries else None
    if teacher_scale:
        for tc_name, tc_overrides in recipe["configs"].items():
            tc_merged = default_cfg.merge(tc_overrides or {})
            if tc_merged.scale == teacher_scale and not tc_merged.auxiliaries:
                tc_map = config_stages.get(tc_name, {})
                for s in ("autoencoder", "curriculum", "normal"):
                    if s == stage and s in tc_map:
                        teacher_asset = tc_map[s]
                        upstream_names.append(teacher_asset)
                        upstream_model_families[teacher_asset] = STAGE_MODEL_MAP[s]
                break                                       # first match wins
```

**Why it's fragile:**

- `merged.auxiliaries[0]` only reads the first KD auxiliary. Multi-teacher KD is
  unrepresented.
- `recipe["configs"].items()` iterates in insertion order. The "first" matching teacher
  depends on recipe YAML key order — rename a config and the student silently relinks.
- Candidate stages are hardcoded as `("autoencoder", "curriculum", "normal")`. A new
  upstream stage silently won't link as a KD teacher.
- Two candidate teacher configs at the same scale → arbitrary choice, no warning.

**Failure mode:** recipe edits that reorder configs or introduce a second same-scale
teacher silently rewire which checkpoint the student loads. The student still trains —
just with the wrong teacher — and the symptom shows up as degraded validation metrics
later.

**What would fix it:** explicit teacher declaration (`teacher_config: baseline_large`
naming a recipe config instead of inferring from `teacher_scale`). Migration cost:
touch every recipe.

### 3. The validation zoo in `resolve.py` — **RESOLVED (Pydantic stage gate)**

**Status:** Cross-field rules live in `graphids/config/cross_field.py` as a
`ValidationRule` table (`_RULES`) and are enforced by
`graphids/config/schemas.py::StageValidation` (Pydantic). The same six
checks are still applied (num_workers within cpus, YAML num_workers within cpus,
GPU partition consistency, curriculum epoch sync, fusion RL batch_size override,
fusion RL YAML batch_size warning). Per-rule unit tests live in
`tests/orchestrate/test_validation_rules.py`.

**Location:** `config/cross_field.py` (rules) + `config/schemas.py`
(Pydantic gate). Structural checks remain in `config/schemas.py`.

Original description — kept for context on why the refactor landed:

This was the largest block of per-stage, per-feature conditionals in the package —
80+ lines of validation rules, each added because something once broke silently:

```python
# Curriculum-specific
if cfg.stage == "curriculum":
    data_max_epochs = data_init.get("max_epochs")
    trainer_max_epochs = trainer.get("max_epochs")
    if data_max_epochs is not None and trainer_max_epochs is not None ...

# Fusion-RL-specific
if cfg.stage == "fusion" and cfg.model_type in ("dqn", "bandit"):
    if "data.init_args.batch_size" in spec.runtime_overrides:
        errors.append(...)

# Resource-level (two paths for the same check)
if resources.num_workers > max_workers: ...
if yaml_workers is not None and int(yaml_workers) > max_workers: ...

# GPU partition consistency
if cfg.stage != "evaluation" and resources.gres:
    if "gpu" not in resources.partition: ...
```

Plus `_convention_errors`:

```python
# LearningRateMonitor requires trainer.logger
for cb in trainer.get("callbacks") or []:
    cp = cb.get("class_path", "")
    if "LearningRateMonitor" in cp and not logger_on: ...

# Null list fields on model init
for fld in ("pool_aggrs", "hidden_dims", "auxiliaries"):
    if fld in model_args and model_args[fld] is None: ...

# Stage monitor conventions (warning, not error)
for ns in ("checkpoint", "early_stopping"):
    for field, expected in zip(("monitor", "mode"), exp): ...
```

**Why it's fragile:**

- **Every rule is a historical scar.** Each protects against a specific past incident.
  There's no shared structure — it's a flat list of conditionals. New failure modes get
  appended forever.
- **Stages are checked by string equality** (`cfg.stage == "curriculum"`). Rename a stage
  and every string literal has to be found and updated.
- **Duplicate check paths.** `num_workers > cpus-1` is validated against both
  `resources.num_workers` and `data_init.get("num_workers")`. Both paths exist because
  the value can come from either source at different merge stages. If merge order changes,
  one becomes redundant but both still run.
- **Warning-vs-error inconsistency.** Stage monitor mismatch → warning. Null list fields
  → error. RL fusion batch_size → error for `runtime_overrides`, warning for YAML. These
  choices are historical, not principled.
- **`_STAGE_MONITORS` lives in `resolve.py`** — it's topology data (which stages optimize
  which metric) in the wrong file. Belongs in `topology.py`.

**What fixed it:** rules-as-data refactor + Pydantic gate. Each rule is a small
object with `name`, `applies`, `check`, `severity`, and StageValidation applies
them consistently during resolution.

### 4. KD JSON blob stringification

**Location:** `resolve.py`.

KD config is a typed `KDEntry` (Pydantic, `config/contracts.py:12`) in the recipe.
Planning extracts it as a dict via `model_dump(exclude_none=True)`. Resolve now passes
the structured payload straight through `jsonnet_tla["auxiliaries"]` via
`graphids.orchestrate.contracts.build_tla_dict`, so the handoff stays typed end-to-end.
There is no JSON/YAML string round-trip and the audit trail records structured values.

## Tier 2 — Smaller but real

### 5. Retry scaling double-dips (`assets.py:77–88`)

On every retry, resources are scaled for both `OUT_OF_MEMORY` and `TIMEOUT` sequentially,
because the retry handler doesn't know which caused the previous failure:

```python
if context.retry_number > 0:
    original = resources
    for reason in ("OUT_OF_MEMORY", "TIMEOUT"):
        resources = scale_resources(resources, reason)
```

After two retries, resources are scaled 4× in both dimensions even if only memory was
ever the problem. Over-allocates on the GPU partition and wastes account time.

**Fix:** read the previous attempt's SLURM terminal state from sacct and scale the
relevant axis only. The state is available — just not plumbed through to the retry
handler.

### 6. `_observe` fires `AssetObservation` from inside the SLURM poll loop (`assets.py:98–104`)

```python
def _observe(slurm_state, jid):
    context.log_event(
        dg.AssetObservation(
            asset_key=context.asset_key,
            metadata={"slurm_state": slurm_state, "job_id": jid},
        )
    )

state, job_id = context.resources.slurm.submit_and_wait(
    ..., on_state=_observe, ...
)
```

`_observe` is passed as a callback into `SubprocessSlurmJobClient.run_training_job`,
which polls sacct and invokes it on state transitions. The callback calls
`context.log_event` from inside that poll loop.

**Why it's fragile:** works in the current `multiprocess_executor` because the poll
loop runs in the same process as the asset body. Assumes `dg.context.log_event` is safe
to call from a polling loop. If we ever swap to an executor that forks or moves the asset
body across processes, events stop appearing silently — no error, just missing data in
the dagster UI.

### 7. `validate.py`'s dedupe skips overrides (`validate.py:74–82`)

```python
chain_key = (
    tuple(cfg.config_files)
    + tuple(sorted(cfg.model_init_overrides.items()))
)
if chain_key in seen:
    continue
```

Two configs with identical YAML file chains and identical `model_init_overrides` but
different `trainer_overrides` / `stage_overrides` / `resource_overrides` / `kd_overrides`
get dedup'd. The second is never validated.

In practice this means `validate` catches every `model_init_overrides` typo but does
**not** catch a typo in `trainer_overrides.trainer.max_epochs` for recipe config #47 if
recipe config #3 had the same YAML chain without that override. The command's name
implies completeness it doesn't deliver.

**Fix:** dedupe on the full `StageConfig` identity (or a hash of all override sources),
not just the CLI chain.

### 8. SLURM error propagation loses the actual error (`assets.py:111–114`)

```python
if state != "COMPLETED":
    log.error("asset_failed", asset=cfg.asset_name, state=state, job_id=job_id)
    raise RuntimeError(f"SLURM job failed: {state}")
```

Debugging a failed run means: read the dagster event log → get the `job_id` → find the
corresponding file in `slurm_logs/` → tail it manually. The traceback / OOM message /
CUDA error is not surfaced in the dagster UI.

`RunRecord` (`core/contracts/run_record.py`) has a `traceback` field that's populated by
the training entrypoint on failure, and the failure path here could read the sidecar and
include it in the `RuntimeError`. Not wired.

## Tier 3 — Smaller

### 9. `_STAGE_MONITORS` in the wrong file

`resolve.py:33` defines `_STAGE_MONITORS = {"autoencoder": ("val_loss", "min"), ...}`.
This is pipeline topology (which stages optimize which metric). It belongs in
`topology.py` alongside `STAGE_MODEL_MAP`. Every other piece of topology was migrated
there already; this one was missed.

### 10. Check dispatch has no registry

After the `multi_asset_check` refactor, `checks._checks()` dispatches by name:

```python
if ckpt_name in selected:
    yield _ckpt_result(...)
if analysis_name in selected:
    yield _analysis_result(...)
```

A third check type requires three edits in sync: (a) an `AssetCheckSpec` in the specs
list, (b) a dispatch branch, (c) a result-builder function. Nothing keeps them aligned.
A tiny registry (`CHECK_TYPES = [(name_prefix, spec_factory, result_factory)]`) would
make this robust at zero runtime cost.

### 11. `build_defs` gets `cfg_lookup` via a one-off comprehension

`component.py:108`:

```python
cfg_lookup = {cfg.asset_name: cfg for cfg in stage_configs}
```

`enumerate_assets` returns a list; `make_asset_checks` wants a dict. Minor, but this is
the kind of seam where a schema change (e.g., two assets colliding on `asset_name`)
would silently lose entries because dict keys collide.

## Recommended next fix

**Tier 1 item #4 — KD JSON blob stringification in `resolve.py`.**

With items #2 and #3 resolved, the remaining Tier 1 item is the 4-layer
serialization round-trip for KD config: `KDEntry` → dict (via `model_dump`) →
JSON string (via `json.dumps`) → YAML string (via `merge_yaml_chain`) →
`KDAuxiliary` TypedDict (via jsonargparse). The audit log stores the JSON
blob verbatim, which is unreadable during diagnostics.

**Fix shape:** make `TrainingSpec.runtime_overrides` typed as `dict[str, Any]`
instead of `dict[str, str]`, then let the structured KD payload flow through
as a dict instead of being stringified. Handle type coercion once inside
`merge_yaml_chain` at the boundary with jsonargparse. Removes the
`json.dumps` + jsonargparse `parse_object` round-trip.

**Why it matters:** the refactor has a dual payoff — the audit log becomes
readable, AND the planning → resolve → transport → runtime path stops losing
type safety in the middle three steps. jsonargparse catches malformed JSON,
but only at the very end, far from whichever layer corrupted the shape.

**Scope:** validate that downstream consumers keep treating `jsonnet_tla` as
typed data (no stringification reintroduced in the CLI path).

---

**Also queued:** Tier 2 items (retry double-dip, `_observe` from poll loop,
`validate.py` dedupe gap, SLURM error propagation) — each is small enough to
land independently once #4 is done.

## Cross-references

- [`orchestration.md`](orchestration.md) — current architecture
- [`3-chain.md`](3-chain.md) — config flow through the 3-handoff pipeline
- [ADR 0009 — Collapse override handoff chain](../decisions/0009-collapse-override-handoffs.md)
