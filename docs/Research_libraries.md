# Library Research

Detailed reports in `docs/<package>_report.md`. Summaries below.

---

## iContract — https://icontract.readthedocs.io/en/latest/usage.html

Design-by-contract decorators (v2.7.3, MIT). `@require` (preconditions),
`@ensure` (postconditions), `@invariant` (class invariants), `@snapshot`
(capture pre-state for postcondition comparison). Auto-generated violation
messages include condition source code and all referenced variable values.

**Key features:** Liskov-correct inheritance via `DBC` base class (preconditions
weaken, postconditions strengthen). `enabled=icontract.SLOW` + `ICONTRACT_SLOW`
env var for expensive dev-only checks. `python -O` disables all contracts globally.
~4 us/check overhead (negligible vs GPU compute).

**Where it fits GraphIDS:** `SLOW`-gated tensor contracts on forward/loss functions
during development — the one place Pydantic cannot reach. Candidates: tensor shape
invariants on encoder input, no-NaN postconditions on loss, `edge_index.max() <
num_nodes` on Data objects. ~4 us/contract at 1000 batches/epoch = 4 ms total.

**Where it doesn't fit:** Config validation (Pydantic already covers this completely).

**Caveats:** Conditions must be side-effect-free (re-executed for error messages).
Tensor comparisons return tensors — must call `.all()`/`.any()`. `DBC` inheritance
is opt-in; forgetting it silently leaks parent contracts. Niche ecosystem (~400 stars).

**Verdict:** Low adoption cost (decorator-only), marginal benefit. Strongest case is
dev-time tensor shape/NaN guards where Pydantic can't reach.

---

## pydantic — https://docs.pydantic.dev/latest/

**Current usage (10 files):** `BaseModel`, `ConfigDict` (frozen/extra), `Field`,
`model_validator(mode="after")`, `field_validator`, `model_validate`/`model_dump`/
`model_dump_json`/`model_validate_json`, `create_model`, `model_rebuild`, `Literal`,
`ClassVar`, `arbitrary_types_allowed`, `AfterValidator` (via `check_in`/`check_all_in`
factories in `graphids/config/validators.py`).

**Handrolled replacements (completed session 38):** `Literal["min","max"]` for monitor
mode, `StageConfig`/`ResourceSpec` promoted to Pydantic, `AwareDatetime` for timestamps.

### Unused features — verified audit (2026-04-08)

#### P0: `pydantic-settings` `BaseSettings` — env var consolidation

Already a dependency (`pyproject.toml:22`), never imported. 19 `os.environ.get`
calls across 14 files with manual type coercion. Verified problems:

| Problem | Scope |
|---------|-------|
| `KD_GAT_LAKE_ROOT` read 9× (only `constants.py:69` is canonical) | `runtime.py`, `staging.py`, `fusion_states.py`, 5 `__init__` defaults |
| 6 `__init__` default-arg uses bake at import time, not call time | `graph.py:93`, `fusion.py:40`, `vgae_module.py:58`, `dgi_module.py:37`, `gat_module.py:49`, `analyzer.py:32` |
| `KD_GAT_SCRATCH` read in 2 files with no shared constant | `cache.py:51`, `staging.py:80` |
| Manual `float()`/`int()`/`.lower()` coercion | `budget.py:37-46`, `resources.py:63`, `definitions.py:32` |

One `BaseSettings` subclass with `env_prefix="KD_GAT_"` would: (a) eliminate all 19
scattered reads, (b) fix the import-time baking bug in 6 `__init__` defaults,
(c) give typed coercion for free (`float`, `int`, `bool`, `Path`).

Centralization modules already exist (`config/constants.py`, `slurm/env.py`) but only
cover 3 vars each — the rest are ad-hoc.

#### P1: Remaining `AfterValidator` / `Literal` gaps

`check_in`/`check_all_in` factories exist and most fields use them. Remaining gaps:

| Location | Current | Fix |
|----------|---------|-----|
| `recipes.py:93-98` `_valid_model_type` | `@field_validator` with `None` guard | `Annotated[str, AfterValidator(check_in(VALID_MODEL_TYPES, ...))] \| None` |
| `recipes.py:100-105` `_stages_exist` | `@model_validator` iterating tuple | `tuple[Annotated[str, AfterValidator(check_in(STAGES, ...))], ...]` |
| `recipes.py:149-155` `_valid_stage_names` | `@field_validator` on dict keys | Keep — dict-key validation has no clean `Annotated` equivalent |
| `contracts/__init__.py:84-87` `normalize_scale` | Hardcoded `{"small", "large"}` | Use `VALID_SCALES` from constants, or delete if callers already Pydantic-validated |
| `monarch/schemas.py:49,51` `conv_type`, `loss_fn` | Plain `str`, no validation | Use `_ConvType`/`_LossFn` Literals from `recipes.py`, or `AfterValidator(check_in(...))` |

Net: ~15 lines deleted across 3 files. `_valid_stage_names` stays (dict-key validation).

#### P2: `@computed_field` — narrower than expected

`PathContext` has 5 properties, `ValidatedConfig` has 2, `ResourceSpec` has 2 — all
invisible to `model_dump()`. However:

- `PathContext` is never `model_dump()`'d. It lives on `ResolvedConfig` (a dataclass)
  and consumers access `.paths.run_dir` directly. **No serialization gap in practice.**
- `ResourceSpec.mem_mb`/`time_minutes` are consumed by internal code, never serialized.
- `ValidatedConfig.checkpoint_monitor`/`checkpoint_mode` are convenience accessors.

`@computed_field` would be correct but the serialization gap doesn't bite today.
Convert if/when these models need to round-trip through JSON (e.g., DuckDB catalog
ingestion). Low urgency.

#### Skip: Discriminated unions

No `Union[...]` annotations exist in Pydantic models. All dispatch is on bare strings
in function args (`build_loss`, `_make_conv`, `build_tla_dict`). Replacing these with
discriminated unions requires promoting `loss_config` dicts into typed Pydantic models
with `type: Literal[...]` fields — a non-trivial schema change for marginal benefit.
Strongest candidate if ever revisited: `loss_config` in `build.py` (already has a
`"type"` key acting as discriminator).

#### Skip: `model_json_schema()`

Never called. Could auto-generate config docs from Pydantic models. Low value — jsonnet
is the user-facing config surface, not Pydantic schemas.

**Priority summary:** ~~P0: `BaseSettings`~~ done (session 39). ~~P1: validator gaps~~ done
(session 39). P2: `@computed_field` (correct but not urgent). Skip: discriminated unions,
`model_json_schema()`.

---

## logging — https://docs.python.org/3/library/logging.html

**Current usage (graphids/log.py, 130 lines, 27 files, 106 call sites):**
`LoggerAdapter` (structured kwargs → `extra`), custom `_JSONFormatter` (JSONL),
`FileHandler` (SLURM logs), `StreamHandler` (stderr), `Filter` (SLURM job ID
injection), single `"graphids"` root logger, idempotent config guard.

**Ignored features:**

- `log.exception()` — never used. `pipeline.py:293-301` manually formats tracebacks via
  `traceback.format_exception()`. `log.exception()` auto-attaches `exc_info`.
- `log.debug()` — zero calls. No way to get verbose tracing in budget/config/staging.
- `RotatingFileHandler` — SLURM JSONL logs grow unbounded. One-line swap.
- `QueueHandler`/`QueueListener` — `ThreadPoolExecutor` in sweep path has thread
  contention on FileHandler lock. Decouples emission from I/O.
- `dictConfig` — all config is imperative. Declarative would be testable/overridable.
- `captureWarnings(True)` — PyTorch/Lightning warnings bypass the logging pipeline.
- `Formatter(defaults=...)` (3.12+) — eliminates `_SlurmFilter` class entirely.

**Handrolled replacements:**

- `_JSONFormatter` iterates `record.__dict__` filtering a 32-entry `_BUILTIN_ATTRS`
  frozenset. Fragile — silently drops `exc_info`/`stack_info`. Must track CPython changes.
- `_SlurmFilter` (6 lines) replaceable by `Formatter(defaults={"slurm_job_id": ""})`.
- Manual traceback formatting in pipeline.py instead of `log.exception()`.

**Priority:** P0: `log.exception()` + fix `_JSONFormatter` exc_info handling, `stacklevel=2`.
P1: add `log.debug()`, swap to `RotatingFileHandler`, `captureWarnings(True)`.
P2: `dictConfig`, `QueueHandler`/`QueueListener` for sweep threads.

---

## tach — https://github.com/tach-org/tach

Rust-powered static import analysis (2.7k stars, MIT). Declares module boundaries in
`tach.toml`: `depends_on` allowlists, `expose` interface patterns, ordered `layers`,
`visibility` controls. `tach check` exits non-zero on violations. Zero runtime cost.

**Key features:** `TYPE_CHECKING` blocks ignored by default. `# tach-ignore(reason)` for
exceptions. `tach sync` auto-discovers deps. `tach test --base main` runs only affected
tests. `tach show --mermaid` visualizes dependency graph. Pre-commit hook built in.
Deprecation tracking (warn-not-fail). VS Code extension. `exact` mode fails on unused deps.

**GraphIDS use cases:**

1. Prevent `orchestrate` → `core` imports (dagster definition-time purity on login node)
2. Enforce `config/` never imports torch (bottom layer, `depends_on = []`)
3. Layer enforcement: `cli > orchestrate > core > config`
4. Interface enforcement: expose only `render_config`, `validate_config` from config/

**Limitations:** `importlib.import_module()` in `instantiate.py` is invisible — needs
`# tach-ignore`. Conditional imports (`try: import torch`) are checked regardless of
runtime reachability. No transitive boundary analysis (per-import, not per-reachability).

**vs import-linter:** Strictly more capable — interface enforcement, deprecation workflows,
affected-test detection, Rust speed. import-linter (991 stars) is simpler but lacks these.

**Verdict:** Strong fit. The project already cares about module boundaries (dagster
definition-time purity rule). `tach check` in CI would enforce this structurally
instead of relying on code review.
