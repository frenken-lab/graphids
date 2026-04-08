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
`ClassVar`, `arbitrary_types_allowed`.

**Ignored features that would help:**

- `pydantic-settings` `BaseSettings` — already a dependency but never imported. 25+
  scattered `os.environ.get("KD_GAT_*")` across 12 files with manual type coercion.
  One `BaseSettings` subclass with `env_prefix="KD_GAT_"` replaces all of them.
- `@computed_field` — `PathContext` has 5 properties invisible to `model_dump()`;
  `ResourceSpec` has 2. These are the exact use case.
- `Annotated[str, AfterValidator(...)]` — 10+ identical set-membership validators in
  `recipes.py` and `monarch/schemas.py` all doing `if v not in SET: raise ValueError`.
- `model_json_schema()` — never called. Could auto-generate config documentation.
- Discriminated unions — manual dispatch where Pydantic could handle it.

**Handrolled replacements:**

- `_MonitorBlock.mode` model_validator checks `mode in ("min","max")` — `Literal["min","max"]` does this natively. 3 lines to delete.
- `StageConfig`/`ResourceSpec` are stdlib dataclasses with manual `to_dict()`/`from_dict()` — Pydantic provides `model_dump()`/`model_validate()` natively.
- `RunRecord` timestamps stored as `str` with manual `.isoformat()` — `AwareDatetime` handles this.
- Contract envelope serialization hand-builds versioned wrappers around `model_dump()`.

**Priority:** P0: `Literal` for mode, `@computed_field`. P1: `BaseSettings` for env vars,
`Annotated` validators. P2: promote dataclasses to Pydantic, `model_json_schema()`.

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
