# GraphIDS Logging Audit

## Section 1: Current Usage

The project uses stdlib `logging` exclusively via `graphids/log.py` (130 lines).

**Features used:**

- **LoggerAdapter** (`_StructuredAdapter`): custom `process()` routes arbitrary kwargs
  into `extra`, enabling structlog-style `log.info("event", key=val)` call sites.
  Used in 27 files, 106 log call sites (all `info`/`warning`/`error`; zero `debug`/`exception`).
- **Custom Formatter** (`_JSONFormatter`): hand-built JSONL formatter iterating
  `record.__dict__`, filtering `_BUILTIN_ATTRS`, serializing to JSON. (`log.py:57-74`)
- **FileHandler**: plain `logging.FileHandler(jsonl_path, mode="a")` for SLURM JSONL output.
  (`log.py:107`)
- **StreamHandler**: stderr with human-readable format for interactive mode. (`log.py:110`)
- **Filter** (`_SlurmFilter`): injects `slurm_job_id` into all records when running
  under SLURM. (`log.py:124-129`)
- **Logger hierarchy**: single `"graphids"` root with `propagate=False`. All child
  loggers inherit via `getLogger(__name__)`. (`log.py:101-103`)
- **Idempotent configuration**: module-level `_configured` guard prevents double setup.
  (`log.py:81,98-99`)
- **Two entry points**: `__main__.py:31` (CLI) and `definitions.py:21` (dagster workers).

## Section 2: Ignored Features

**High value, directly applicable:**

1. **`log.exception()`** -- never used. Error handlers in `pipeline.py:293-301` manually
   format tracebacks via `traceback.format_exception()` and pass as a kwarg
   (`traceback="".join(last_tb)`). `log.exception()` auto-attaches `exc_info` and the
   formatter handles it. Same pattern in `actors.py:250-265` where exceptions are
   caught and only `str(exc)` is logged.

2. **`log.debug()`** -- zero calls across 106 log sites. Budget probing (`budget.py`),
   config resolution (`resolver.py`), data staging (`staging.py`) all have multi-step
   logic that would benefit from debug-level tracing without polluting INFO output.

3. **`RotatingFileHandler`** -- SLURM orchestrator JSONL files grow unbounded
   (`FileHandler(mode="a")`). Long sweeps with many chains produce large logs.
   `RotatingFileHandler(maxBytes=10_000_000, backupCount=3)` prevents unbounded growth.

4. **`dictConfig`** -- all configuration is imperative code in `configure_logging()`.
   `dictConfig` would make the dual-mode setup (JSONL vs stderr) declarative, testable,
   and overridable without code changes. Supports `ext://sys.stderr`, formatter classes
   via `()` factory, and `disable_existing_loggers=False`.

5. **`QueueHandler`/`QueueListener`** -- `monarch/pipeline.py:213` runs chains in
   `ThreadPoolExecutor`. Each thread's log calls hit the same `FileHandler` concurrently.
   `FileHandler.emit()` acquires a thread lock, serializing all logging I/O.
   `QueueHandler` + `QueueListener` would decouple log emission from I/O, keeping worker
   threads non-blocking. Even more relevant if monarch actors run across processes.

6. **Formatter `defaults` parameter** (Python 3.12+) -- the human-readable format string
   (`log.py:112`) doesn't include structured fields. `defaults={"slurm_job_id": ""}` would
   let the format string reference `%(slurm_job_id)s` without KeyError when not under SLURM,
   eliminating the need for the `_SlurmFilter` class entirely.

7. **`logging.captureWarnings(True)`** -- PyTorch and Lightning emit `warnings.warn()` for
   deprecations and performance hints. These go to stderr unstructured. Capturing them into
   the logging pipeline routes them through the same JSONL/stderr handler pair.

**Lower priority but relevant:**

8. **Handler-level filtering** -- currently one handler per config. If both JSONL and stderr
   were active simultaneously (e.g., tee verbose to file, summary to console), per-handler
   `setLevel()` or filters would route by severity.

9. **`stacklevel` parameter** -- the adapter's `process()` adds one frame. Stdlib logging
   uses `stacklevel` to report the correct caller file:line. Currently not passed, so
   `_JSONFormatter`'s `%(filename)s:%(lineno)d` (if used) would point to the adapter, not
   the call site.

## Section 3: Handrolled Replacements

1. **JSONL formatter (`_JSONFormatter`, log.py:57-74)** -- iterates `record.__dict__`,
   filters against `_BUILTIN_ATTRS` frozenset, tries `json.dumps()` per value. This is a
   manual reimplementation of what `python-json-logger` (or a minimal stdlib subclass with
   `defaults`) does. The `_BUILTIN_ATTRS` set (32 entries) must be kept in sync with
   CPython's `LogRecord.__init__`; any new attribute added in a Python upgrade silently
   leaks into JSONL output or gets filtered when it shouldn't.

2. **Traceback formatting (pipeline.py:283-300)** -- `traceback.format_exception(exc)` is
   called manually and passed as `traceback=` kwarg. Stdlib's `exc_info=True` or
   `log.exception()` does this automatically and the formatter handles serialization
   (including into JSON via `formatException()`).

3. **Idempotent configuration guard (log.py:81,98-99)** -- module-level `_configured` bool.
   `dictConfig` with `incremental=False` is inherently idempotent (replaces handlers).
   Or: `logging.root.handlers` check is the stdlib pattern for "already configured."

4. **SLURM context injection via Filter (log.py:124-129)** -- a 6-line inner class that
   sets `record.slurm_job_id`. Stdlib `LogRecordFactory` (via `setLogRecordFactory()`) or
   `Formatter(defaults={"slurm_job_id": ...})` achieves the same with less code. The
   factory approach is especially clean: one `setLogRecordFactory()` call replaces the
   class definition + `addFilter()` call.

5. **Manual log file path construction (definitions.py:21)** -- `f"{SLURM_LOG_DIR}/
   orchestrator_{_slurm_job}.jsonl"` is built inline. This is fine, but a `dictConfig`
   approach with `cfg://` references would centralize all path logic.

## Section 4: Recommendations (Prioritized)

**P0 -- Fix correctness / silent issues:**

1. Use `log.exception()` instead of manual traceback formatting in `pipeline.py:293-301`.
   Removes 4 lines, gains proper `exc_info` handling in JSON output. The `_JSONFormatter`
   should call `self.formatException(record.exc_info)` and include it in the JSON dict
   (currently silently drops exception info since `exc_info` is in `_BUILTIN_ATTRS`).

2. Add `stacklevel=2` to `_StructuredAdapter.process()` return or override `log()` to
   pass it, so `%(filename)s:%(lineno)d` in log records points to the actual call site.

**P1 -- Low-effort wins (< 30 min each):**

3. Add `log.debug()` calls in budget probing, config resolution, and data staging.
   Support `--verbose` / `-v` CLI flag that sets level to `DEBUG`.

4. Replace `FileHandler` with `RotatingFileHandler(maxBytes=50_000_000, backupCount=3)`
   for SLURM JSONL logs. One-line change in `configure_logging()`.

5. Add `logging.captureWarnings(True)` to `configure_logging()`. One line, routes
   PyTorch/Lightning warnings through the structured logging pipeline.

**P2 -- Architectural improvements (1-2 hours):**

6. Replace imperative `configure_logging()` with `dictConfig`. Define two config dicts
   (SLURM mode, interactive mode) and select based on `slurm_job_id()`. Makes the
   logging setup testable and overridable.

7. Add `QueueHandler`/`QueueListener` for the sweep ThreadPoolExecutor path in
   `monarch/pipeline.py`. Prevents thread contention on `FileHandler`'s lock during
   parallel chain execution.

**P3 -- Nice to have:**

8. Use `Formatter(defaults={"slurm_job_id": ""})` (Python 3.12+) to eliminate the
   `_SlurmFilter` class. Requires confirming Python 3.12 is the minimum (it is per
   `pyproject.toml`).

9. Harden `_JSONFormatter` by calling `self.formatException()` for `exc_info` and
   `self.formatStack()` for `stack_info`, rather than silently filtering them via
   `_BUILTIN_ATTRS`.
