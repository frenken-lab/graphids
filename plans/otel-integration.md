# Plan: OpenTelemetry as Unified Observability Layer

> Created: 2026-04-08 | Research: `docs/opentelemetry_research.md`

## Goal

OTel as first-class dep. `traces.jsonl` replaces `run_record.json`.
No wrapper functions — use OTel global API directly.

## Verified SDK Facts

| Fact | Source |
|------|--------|
| `ConsoleSpanExporter(out=file)` — default formatter: `span.to_json() + linesep` | [SDK source](https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/trace/export/__init__.py) |
| `ConsoleMetricExporter(out=file)` — default: `metrics_data.to_json() + linesep` | [SDK source](https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/metrics/_internal/export/__init__.py) |
| `ConsoleLogRecordExporter(out=file)` — default: `record.to_json() + linesep` | [SDK source](https://github.com/open-telemetry/opentelemetry-python/blob/main/opentelemetry-sdk/src/opentelemetry/sdk/_logs/_internal/export/__init__.py) |
| Wandb Weave OTLP endpoint: `trace.wandb.ai/otel/v1/traces`, auth via `wandb-api-key` header | [Wandb docs](https://docs.wandb.ai/weave/guides/tracking/otel) |
| `TracerProvider.add_span_processor()` works after `set_tracer_provider()` at runtime; mypy flags it (API type lacks method) — keep SDK reference | [Issue #3713](https://github.com/open-telemetry/opentelemetry-python/issues/3713) |
| `MeterProvider` readers are constructor-only — cannot add after creation | SDK docs |
| `otel-file-exporter` requires Python 3.13+ — unusable on OSC 3.12 | [PyPI](https://pypi.org/project/otel-file-exporter/) |
| OSC login node → `trace.wandb.ai`: HTTP 200. Compute node: unverified | curl test this session |

Zero custom exporters needed. Zero wrapper functions.

---

## Write Path Migration

| # | Current | Replacement |
|---|---------|-------------|
| 1 | SLURM stdout/stderr (`submit.sh`) | Unchanged |
| 2 | Orchestrator JSONL (`_JSONFormatter` + `FileHandler`, `log.py:105`) | OTel `LoggingHandler` → `ConsoleLogRecordExporter(out=logfile)` |
| 3 | Interactive stderr (`log.py:110`) | `ConsoleLogRecordExporter(out=sys.stderr)` |
| 4 | `run_record.json` sidecar (`RunRecordCallback`, `base.py:368`) | **Delete.** `traces.jsonl` IS the run record — span attributes carry identity, status, metrics, timing |
| 5 | `config_snapshot.json` (`io.py:80`) | Unchanged — reproducibility artifact |
| 6 | `resource_profile.csv` (`base.py:323`, BROKEN) | **Delete.** OTel metrics in callback |
| 7 | wandb metrics (`WandbLogger`) | `OTelTrainingLogger` → `metrics.jsonl` + Weave via OTLP |
| 8 | wandb config push (`instantiate.py:142`) | Span attributes → Weave |
| 9 | wandb system metrics (pynvml) | OTel gauges in callback |
| 10 | CSVLogger (`instantiate.py:87`) | `OTelTrainingLogger` |
| 11 | DeviceStatsMonitor (`defaults.libsonnet:28`) | OTel callback |
| 12 | sacct epilog (`_epilog.sh:6`) | Unchanged |
| 13 | `finalize_run_record` (`finalize.py`) | **Delete.** Span has start/end timestamps. Phase markers are separate files. |
| 14 | DuckDB catalog (`catalog.py`) | **Rewrite.** Query `traces.jsonl` (nested OTel schema) |
| 15 | budget_calibration.csv | Unchanged |
| 16 | Checkpoints | Unchanged |
| 17 | WANDB_DIR mkdir (`_preamble.sh:33`) | **Delete** |

---

## Obsolete Code (delete)

| File | What | Lines |
|------|------|-------|
| `graphids/core/run_record.py` | Entire file — `RunRecord` schema | 54 |
| `graphids/core/models/base.py:306-451` | `ResourceProfileCallback` + `RunRecordCallback` | 145 |
| `graphids/orchestrate/ops/finalize.py` | Entire file — patches sidecar that no longer exists | 53 |
| `graphids/log.py` | `_JSONFormatter`, `_SlurmFilter`, `_BUILTIN_ATTRS`, all handler setup | ~70 |
| `graphids/core/io.py` | `write_run_record`, `read_run_record` (keep `snapshot_config`, `_atomic_write_text`) | ~30 |
| `graphids/instantiate.py` | `_WANDB_WRITE_DIR` (line 24), WandbLogger/CSVLogger patching (71-90), wandb config push (142-148) | ~30 |
| `graphids/cli/_orchestrate.py` | `_finalize-record` command | ~10 |
| `graphids/config/constants.py` | `RUN_RECORD_FILENAME` | 1 |
| `graphids/orchestrate/actors.py:183-189` | `finalize_run_record` call | 7 |
| `graphids/orchestrate/ops/__init__.py` | `finalize_run_record` export | 2 |
| `configs/_lib/defaults.libsonnet` | `device_stats`, `resource_profile`, `run_record` entries | ~12 |
| `scripts/slurm/_preamble.sh` | `WANDB_DIR` mkdir + export | ~3 |

**Total deleted: ~415 lines**

---

## New Code

### `__main__.py` — Phase A setup (~30 lines, replaces `configure_logging()`)

Providers created, wandb OTLP wired, logging bridge attached. No file
exporters — `run_dir` unknown until config resolution.

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogRecordExporter
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk.resources import Resource

resource = Resource.create({
    "service.name": "graphids",
    "slurm.job_id": os.environ.get("SLURM_JOB_ID", ""),
    "wandb.entity": os.environ.get("WANDB_ENTITY", ""),
    "wandb.project": os.environ.get("WANDB_PROJECT", "graphids"),
})
# Keep SDK references (get_tracer_provider() returns API type without add_span_processor)
_tp = TracerProvider(resource=resource)
if os.environ.get("WANDB_API_KEY"):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
        endpoint="https://trace.wandb.ai/otel/v1/traces",
        headers={"wandb-api-key": os.environ["WANDB_API_KEY"]},
    )))
trace.set_tracer_provider(_tp)
metrics.set_meter_provider(MeterProvider(resource=resource))
lp = LoggerProvider(resource=resource)
lp.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogRecordExporter(out=sys.stderr)))
set_logger_provider(lp)
logging.getLogger("graphids").addHandler(LoggingHandler(logger_provider=lp))
atexit.register(lambda: (_tp.shutdown(), lp.shutdown()))
```

### `train_entrypoint.py` — Phase B file exporters (~15 lines in `_execute`)

```python
run_dir = Path(run.trainer.default_root_dir)
_tp.add_span_processor(SimpleSpanProcessor(
    ConsoleSpanExporter(out=open(run_dir / "traces.jsonl", "a"))
))
# MeterProvider readers are constructor-only — replace provider
mp = MeterProvider(resource=_tp.resource, metric_readers=[
    PeriodicExportingMetricReader(
        ConsoleMetricExporter(out=open(run_dir / "metrics.jsonl", "a")),
        export_interval_millis=10_000,
    )
])
metrics.set_meter_provider(mp)
```

`_tp` imported from `__main__`. Monarch actors: same Phase A in
`__initialize__`, Phase B when stage config resolves `run_dir`.

### `graphids/core/monitoring.py` — callback + logger (~120 lines)

```python
class OTelTrainingCallback(pl.Callback):
    """Replaces ResourceProfileCallback + RunRecordCallback + DeviceStatsMonitor."""
    def on_fit_start(self, trainer, pl_module):
        self._span = trace.get_tracer(__name__).start_span("training.fit")
        # identity, config, model class as span attributes

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        meter = metrics.get_meter(__name__)
        # loss histogram, VRAM gauges, batch timing — direct OTel API

    def on_fit_end(self, trainer, pl_module):
        # callback_metrics as span attributes, status OK, end span

    def on_exception(self, trainer, pl_module, exception):
        # record exception, status ERROR, end span


class OTelTrainingLogger(lightning.pytorch.loggers.Logger):
    """Replaces WandbLogger + CSVLogger. Captures self.log() → OTel metrics."""
    def __init__(self):
        super().__init__()
        self._meter = metrics.get_meter(__name__)
        self._instruments: dict[str, Histogram] = {}

    def log_metrics(self, metrics_dict, step):
        for name, value in metrics_dict.items():
            if name not in self._instruments:
                self._instruments[name] = self._meter.create_histogram(name)
            self._instruments[name].record(value, {"step": step})

    def log_hyperparams(self, params):
        span = trace.get_current_span()
        for k, v in _flatten(params).items():
            span.set_attribute(f"hparam.{k}", _coerce(v))
```

### `graphids/log.py` — adapter only (~20 lines)

```python
"""Structured logging adapter for log.info("event", key=val) call sites."""
import logging
from typing import Any

class _StructuredAdapter(logging.LoggerAdapter):
    """Routes arbitrary kwargs into extra. Load-bearing for 89 call sites —
    stdlib Logger.info() raises TypeError on arbitrary kwargs."""
    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        extra: dict[str, Any] = {**self.extra, **kwargs.pop("extra", {})}
        for k in list(kwargs):
            if k not in ("exc_info", "stack_info", "stacklevel"):
                extra[k] = kwargs.pop(k)
        kwargs["extra"] = extra
        return msg, kwargs

def get_logger(name: str | None = None) -> _StructuredAdapter:
    return _StructuredAdapter(logging.getLogger(name or "graphids"), {})
```

### Catalog + status rewrites (~30 lines changed)

`traces.jsonl` uses OTel span schema — nested, not flat:
```sql
CREATE OR REPLACE TABLE runs AS
SELECT
    json_extract_string(attributes, '$.ml.stage') AS stage,
    json_extract_string(attributes, '$.ml.identity_hash') AS identity_hash,
    json_extract_string(status, '$.status_code') AS status_code,
    start_time, end_time, ...
FROM read_json_auto(?, format='newline_delimited')
WHERE name = 'training.fit'
```

`status.py` maps `OK/ERROR/UNSET` (OTel) not `completed/failed/started`.

**Total added: ~215 lines**

---

## Dependencies

```diff
# pyproject.toml
+ "opentelemetry-api>=1.25",
+ "opentelemetry-sdk>=1.25",
+ "opentelemetry-exporter-otlp-proto-http>=1.25",
- "wandb>=0.25.1",
```

## Net Impact

| | Lines |
|---|-------|
| Deleted | ~415 |
| Added | ~215 |
| **Net** | **-200** |

Swap 1 dep (wandb) for 3 (otel-api, otel-sdk, otel-exporter-otlp-http).

---

## Implementation Hazards

1. **`_StructuredAdapter` is load-bearing.** 89 call sites use `log.info("event", key=val)`. Do not delete.

2. **`run_dir` unknown at process start.** Phase A (providers, wandb, logging bridge) in `__main__.py`. Phase B (file exporters) in `_execute()` after config resolution.

3. **Monarch actors bypass `__main__.py`.** Need Phase A setup in actor `__initialize__`.

4. **`get_tracer_provider()` returns API type** — no `add_span_processor`. Keep SDK `_tp` reference for Phase B.

5. **`MeterProvider` readers are constructor-only.** Phase B creates new `MeterProvider`, replaces global.

6. **`create_histogram()` must be cached.** `OTelTrainingLogger._instruments` dict, one call per metric name.

7. **`traces.jsonl` schema ≠ `run_record.json` schema.** Catalog rewrite extracts from nested `attributes` dict.

8. **OSC compute → internet unverified.** `BatchSpanProcessor` silently drops on timeout (training won't hang). Verify with `curl` from SLURM job.

9. **File handle flush on SIGUSR1.** `atexit.register` handles normal exit. Wire `_tp.shutdown()` into SIGUSR1 handler for SLURM preemption.

## Implementation Protocol

### Rules

- **I write all code changes myself.** No delegating edits to subagents.
  Subagents research and verify — they don't write production code.
- **Foreground subagents only.** Background agents hang and you lose trust.
  If a subagent takes >2 min, kill it and do the work directly.
- **Verify after every batch.** Each batch ends with a gate check before
  the next batch starts. No "I'll check later."
- **Read before edit.** Re-read every file before modifying — no edits
  from stale memory of what a file contains.
- **Grep after deletions.** Every deleted symbol gets a repo-wide grep
  to find remaining references. Dangling imports = broken code.

### Batches and Gates

**Batch 1: Foundation** (must pass gate before anything else)
- `pyproject.toml` — add deps, remove wandb
- `uv sync`
- `python -c "from opentelemetry import trace; print('ok')"`

Gate 1: OTel importable. `ruff check graphids/`.

**Batch 2: New code** (no deletions yet — old + new coexist)
- `core/monitoring.py` — `OTelTrainingCallback` + `OTelTrainingLogger`
- `__main__.py` — Phase A setup (replaces `configure_logging()` call)
- `train_entrypoint.py` — Phase B file exporters in `_execute()`
- `defaults.libsonnet` + stage jsonnet — wire new callback + logger

Gate 2: `python -c "from graphids.core.monitoring import OTelTrainingCallback, OTelTrainingLogger"`.
`ruff check graphids/`. `jsonnet configs/stages/autoencoder.jsonnet` renders.
Old callbacks still importable (not deleted yet — that's Batch 3).

**Batch 3: Deletions** (one at a time, grep after each)
- `log.py` → strip to adapter-only
- `base.py` → delete ResourceProfileCallback + RunRecordCallback
- `run_record.py` → delete entire file
- `io.py` → delete sidecar functions
- `finalize.py` → delete entire file
- `instantiate.py` → delete wandb patching
- `constants.py` → delete RUN_RECORD_FILENAME
- `actors.py` → delete finalize_run_record call
- `_orchestrate.py` → delete _finalize-record command
- `ops/__init__.py` → remove finalize_run_record export
- `defaults.libsonnet` → remove old callback entries
- `_preamble.sh` → remove WANDB_DIR

After EACH deletion: `grep -r "<deleted_symbol>" graphids/ configs/ scripts/`
to catch dangling references. Fix before next deletion.

Gate 3: `ruff check graphids/`. `python -c "import graphids"` (no import
errors). `python -m graphids --help` (CLI loads). All jsonnet stages render.

**Batch 4: Rewrites**
- `catalog.py` — read traces.jsonl with nested OTel span schema
- `status.py` — map OK/ERROR/UNSET instead of completed/failed/started
- Monarch actor — Phase A in `__initialize__`, Phase B at stage start

Gate 4: `python -c "from graphids.orchestrate.ops.catalog import rebuild_catalog"`.
`ruff check graphids/`.

**Batch 5: Docs + final verification**
- `docs/reference/observability.md` — rewrite
- SLURM: `curl --max-time 5 https://trace.wandb.ai` from compute node
- SLURM: `fast_dev_run` → check traces/metrics/logs JSONL
- `rebuild-catalog` + `pipeline-status` on new format

### What subagents do (and don't do)

| Task | Who | Why |
|------|-----|-----|
| Write/edit code | Me | Consistency, no drift |
| `ruff check` after batches | Me (Bash) | Fast, deterministic |
| Grep for dangling refs after deletions | Me (Grep tool) | Must see results myself |
| Verify jsonnet renders | Me (Bash) | Must see errors myself |
| Research if I hit an unexpected OTel API question | Subagent (foreground, capped) | Bounded question, verifiable answer |
| Review final diff before commit | Subagent (reviewer) | Second pair of eyes on the complete change |

No subagent writes code. No subagent runs in background on the critical
path. If I spawn one and it hasn't returned in 2 minutes, I move on
without it.
