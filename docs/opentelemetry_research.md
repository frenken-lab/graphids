# OpenTelemetry as Unified Observability Layer for GraphIDS

Research into replacing the current multi-system observability stack with
OpenTelemetry (OTel) as the single provider for logging, metrics, traces,
artifact tracking, and experiment metadata.

## 1. Current State

GraphIDS observability is split across five disconnected systems:

| Signal | Current Implementation | Location |
|--------|----------------------|----------|
| **Structured logging** | stdlib `logging` + `_StructuredAdapter` + `_JSONFormatter` | `graphids/log.py` (130 lines) |
| **Experiment tracking** | Lightning `WandbLogger` + `CSVLogger`, patched at instantiation | `graphids/instantiate.py:24-150` |
| **Run metadata** | `RunRecord` Pydantic model, atomic JSON sidecar | `graphids/core/run_record.py`, `core/io.py` |
| **Resource profiling** | `ResourceProfileCallback` (CSV) — **incomplete: writes header, never writes rows** | `graphids/core/models/base.py:323-366` |
| **Catalog** | DuckDB rebuilt from `run_record.json` sidecars | `graphids/orchestrate/ops/catalog.py` |

### Problems with this architecture

1. **No correlation.** Logs, metrics, and run records have no shared identity.
   A SLURM job ID in a log line cannot be traced to the metrics that job produced
   without manual cross-referencing.

2. **Five write paths.** Each system has its own serialization, file format,
   and error handling. `run_record.json` uses atomic temp+fsync+rename.
   `resource_profile.csv` uses a CSV writer. Wandb uses its own sync daemon.
   Stdlib logging uses a FileHandler. DuckDB catalog is rebuilt from sidecars.

3. **Incomplete coverage.** `ResourceProfileCallback` opens a CSV file but
   has no `on_train_batch_end` hook — it never writes rows. Device stats
   (VRAM, RSS, step timing) are not actually recorded anywhere.

4. **Wandb dependency for metrics.** Training metrics flow through Wandb's
   proprietary sync protocol. Offline analysis requires either Wandb's API
   or parsing CSV logger output.

5. **No cross-process tracing.** The pipeline runs across SLURM jobs
   (autoencoder -> supervised -> fusion), but there's no trace continuity
   between stages. `run_record.json` captures per-stage status but not
   the causal chain.

## 2. OpenTelemetry Overview

OpenTelemetry is a CNCF-graduated vendor-neutral observability framework
providing three signal types through a unified API:

- **Traces** — distributed call trees made of *spans* with attributes,
  events, and parent-child relationships
- **Metrics** — counters, histograms, gauges with dimensional attributes
- **Logs** — bridges existing `logging.Logger` calls into the OTel pipeline

All three share a `Resource` (identity metadata attached to every signal),
use the same exporter pattern, and support context propagation across
threads and processes.

### Python SDK packages

| Package | Purpose |
|---------|---------|
| `opentelemetry-api` | Stable interfaces (Tracer, Meter, Logger) |
| `opentelemetry-sdk` | Implementations (TracerProvider, MeterProvider, LoggerProvider) |
| `opentelemetry-semantic-conventions` | Standard attribute name constants |
| `opentelemetry-exporter-otlp-proto-grpc` | OTLP gRPC exporter (all signals) |

### Key APIs

**Traces:**

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("graphids.training")

with tracer.start_as_current_span("training.epoch") as span:
    span.set_attribute("ml.epoch", epoch)
    span.add_event("checkpoint.saved", {"path": str(ckpt_path)})
```

**Metrics:**

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider

meter = metrics.get_meter("graphids.training")
loss_histogram = meter.create_histogram("ml.train.loss", unit="1")
batch_counter = meter.create_counter("ml.batches_processed")

loss_histogram.record(0.342, {"ml.stage": "supervised", "ml.epoch": 5})
batch_counter.add(1, {"ml.stage": "supervised"})
```

**Logs (bridge, not replacement):**

```python
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

log_provider = LoggerProvider(resource=resource)
log_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

# Bridge stdlib logging → OTel (existing log.info() calls work unchanged)
handler = LoggingHandler(logger_provider=log_provider)
logging.getLogger("graphids").addHandler(handler)
```

When a log call happens inside an active span, OTel automatically injects
`trace_id` and `span_id` into the log record — giving free log-trace
correlation.

### Context propagation across processes

OTel uses W3C TraceContext by default. For cross-SLURM-job trace continuity:

```python
from opentelemetry import propagate

# In stage N: serialize current context
carrier = {}
propagate.inject(carrier)
# Write carrier to run_record.json or env var

# In stage N+1: restore context
ctx = propagate.extract(carrier)
with tracer.start_as_current_span("fusion.stage", context=ctx):
    ...  # becomes a child of the upstream stage's span
```

This replaces the manual `run_record.json` linking with automatic trace
trees spanning the full pipeline.

### Custom exporters

OTel's exporter interface is minimal — three methods:

```python
class SpanExporter:
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult: ...
    def shutdown(self) -> None: ...
    def force_flush(self, timeout_millis: int) -> bool: ...
```

Same pattern for `MetricExporter` and `LogExporter`. This means we can
write exporters that emit to any backend: JSONL files, DuckDB, Parquet,
run_record.json sidecars — without changing instrumentation code.

## 3. Mock Implementation Analysis

The `open_telemetry_mock/` directory contains a template implementation
by a prior Claude session. Key design decisions:

### `PipelineLogger` — single class, three OTel pillars

`logger.py` defines a `PipelineLogger` that initializes all three OTel
providers (TracerProvider, MeterProvider, LoggerProvider) sharing one
`Resource`. The `LoggerConfig` Pydantic model selects a backend via
`ExporterBackend` enum (`CONSOLE`, `FILE`, `OTLP_GRPC`, `MEMORY`).

Changing `backend=` in config is the only change needed to redirect all
three signal types simultaneously — no code changes in instrumentation.

### Pydantic signal schemas

Seven typed schemas (`DatasetMeta`, `TrainingMetrics`, `DeviceStats`,
`EvalMetrics`, `ArtifactRecord`, `ProfileEvent`, `ExperimentConfig`)
validate data before any OTel call. Every `log_*` method accepts a typed
object — invalid data fails at Pydantic validation, not at the exporter.

### Dual-write pattern: metrics + span events

Every metric recording also fires a `span.add_event()`. This means
metric values are queryable from trace data even without a metrics
backend configured. Trade-off: more data written, but eliminates the
"metrics backend required" dependency for analysis.

### Gauge via UpDownCounter + delta tracking

OTel has no settable gauge instrument. The mock tracks last-value per
metric name and emits deltas via `UpDownCounter`. Additionally records
each absolute value as a span event for queryability.

### Artifact tracking: pointer model

`log_artifact()` records a URI string (not bytes) as a span event of
kind `artifact.<type>`. Artifacts appear in the trace timeline alongside
the spans that produced them.

### Cross-process context via carrier dict

`inject_context()` serializes the current span's trace/span IDs into a
`dict[str, str]` (W3C traceparent format). `extract_context()` deserializes
it. The `stage()` context manager passes the restored context to
`start_as_current_span`, making the new span a child of the upstream span.

### File exporters: three JSONL files

`_FileSpanExporter`, `_FileMetricExporter`, `_FileLogExporter` each
write to a separate JSONL file (`traces.jsonl`, `metrics.jsonl`,
`logs.jsonl`). Simple and queryable, though `_FileMetricExporter` uses
`str(metric.data)` instead of structured JSON — needs fixing.

### Lightning callback adapter

`LightningPipelineLogger` reads `trainer.callback_metrics` to capture
any metrics the LightningModule logged via `self.log()`. Pulls LR from
`trainer.optimizers[0].param_groups[0]["lr"]`. Reads
`trainer.checkpoint_callback.best_model_path` for artifact tracking.

### Caveats in the mock

1. `_FileMetricExporter.export` serializes `metric.data` as `str()` —
   not structured JSON. Would need proper serialization.
2. `MEMORY` backend only captures spans, not metrics or logs.
3. `example_usage.py` imports from `pipeline_logger.logger` but the
   directory is `open_telemetry_mock/` — not runnable without rename.
4. `extract_context` stores `self._remote_ctx` but the token from
   `trace.use_span()` is never detached. Works because `stage()` passes
   `ctx=` directly to `start_as_current_span`.
5. `torch>=2.0.0` listed without upper bound in requirements.txt.

## 4. What Would Change

### Replaced

| Current | Replacement |
|---------|-------------|
| `graphids/log.py` (130 lines) | OTel `LoggingHandler` bridge — stdlib `logging` calls flow through OTel automatically. `_StructuredAdapter` can stay; `_JSONFormatter` and `_SlurmFilter` become unnecessary (OTel handles formatting and attribute injection). |
| `WandbLogger` | OTel metrics + custom `WandbExporter` (optional, for W&B UI). Or drop Wandb entirely and use JSONL/DuckDB for analysis. |
| `CSVLogger` | OTel `_FileMetricExporter` (JSONL, not CSV). |
| `ResourceProfileCallback` | OTel histogram instruments for VRAM/RSS/step timing, recorded in `on_train_batch_end`. Device stats become first-class metrics, not a broken CSV. |
| `RunRecordCallback` + `run_record.json` | OTel span attributes on the pipeline/stage span. Run status, timing, metrics, SLURM context all become span attributes. A custom `RunRecordExporter` can still write the JSON sidecar if needed for backward compatibility with the DuckDB catalog. |
| Manual cross-stage linking | OTel trace context propagation. Pipeline spans across SLURM jobs form a single trace tree. |

### Preserved

| Component | Reason |
|-----------|--------|
| `RunRecord` Pydantic model | Schema definition is still useful for validation, even if the write path changes. |
| DuckDB catalog | Consumers query it. Could read from OTel JSONL or run_record.json — either way, the catalog rebuild logic stays. Alternatively, `duckdb-otlp` extension can query traces.jsonl directly. |
| `ModelCheckpoint` / `EarlyStopping` | Lightning framework callbacks — not observability. |
| `graphids.config.*` | Config system is orthogonal to observability. |
| `torchmetrics` | Metric computation stays in Lightning. OTel captures the computed values, not the computation. |

### New

| Component | Purpose |
|-----------|---------|
| `graphids/telemetry/provider.py` | Single setup function: creates Resource, TracerProvider, MeterProvider, LoggerProvider with shared identity. Selects exporters based on config (FILE for dev, OTLP for production, MEMORY for tests). |
| `graphids/telemetry/instruments.py` | Pre-defined OTel instruments (histograms for loss/timing, counters for batches/epochs, gauges for VRAM). Avoids instrument creation scattered across files. |
| `graphids/telemetry/exporters.py` | Custom exporters: `JsonlFileSpanExporter`, `JsonlFileMetricExporter`, `RunRecordSpanExporter` (writes run_record.json from span data). |
| `graphids/telemetry/callback.py` | Single Lightning callback replacing `ResourceProfileCallback` + `RunRecordCallback`. Delegates to OTel instruments. |
| `graphids/telemetry/context.py` | Helpers for cross-SLURM-job context propagation (inject into run_record.json, extract on stage start). |

## 5. Mapping to GraphIDS Pipeline

### Trace hierarchy

```
pipeline (trace root)
├── stage: autoencoder
│   ├── data.setup (DataModule.prepare_data + setup)
│   ├── training.fit
│   │   ├── epoch.0
│   │   │   ├── batch.0 ... batch.N
│   │   │   └── validation
│   │   ├── epoch.1 ...
│   │   └── checkpoint.saved (event)
│   ├── testing
│   └── analysis (VGAE latent space viz, reconstruction plots)
├── stage: supervised
│   ├── link: autoencoder (span link to upstream stage)
│   ├── data.setup
│   ├── training.fit
│   │   └── ... (same structure)
│   └── testing
└── stage: fusion
    ├── link: autoencoder + supervised
    ├── training.fit
    └── testing
```

### Resource attributes (shared across all signals)

```python
resource = Resource.create({
    "service.name": "graphids",
    "service.version": git_sha,
    "ml.pipeline.name": "kd-gat",
    "ml.dataset": "hcrl_sa",
    "ml.model.family": "gat",
    "ml.model.scale": "small",
    "ml.stage": "supervised",
    "ml.identity_hash": "a1b2c3d4",
    "ml.seed": 42,
    "slurm.job_id": os.environ.get("SLURM_JOB_ID", ""),
    "slurm.partition": os.environ.get("SLURM_JOB_PARTITION", ""),
})
```

### Metric instruments

```python
meter = metrics.get_meter("graphids.training")

# Training metrics
train_loss    = meter.create_histogram("ml.train.loss")
val_loss      = meter.create_histogram("ml.val.loss")
learning_rate = meter.create_gauge("ml.learning_rate")       # OTel 1.x gauge
epoch_dur     = meter.create_histogram("ml.epoch.duration_s", unit="s")
batch_dur     = meter.create_histogram("ml.batch.duration_s", unit="s")

# Resource metrics
cuda_alloc    = meter.create_gauge("ml.cuda.allocated_mb", unit="MiB")
cuda_reserved = meter.create_gauge("ml.cuda.reserved_mb", unit="MiB")
host_rss      = meter.create_gauge("ml.host.rss_mb", unit="MiB")

# Throughput
batches       = meter.create_counter("ml.batches.total")
samples       = meter.create_counter("ml.samples.total")
```

### Artifact tracking as span events

```python
# Checkpoint saved
span.add_event("artifact.checkpoint", {
    "artifact.uri": str(ckpt_path),
    "artifact.type": "model_checkpoint",
    "ml.val.loss": best_val_loss,
    "ml.epoch": current_epoch,
})

# Analysis output
span.add_event("artifact.analysis", {
    "artifact.uri": str(plot_path),
    "artifact.type": "latent_space_plot",
    "artifact.format": "png",
})

# Config snapshot
span.add_event("artifact.config", {
    "artifact.uri": str(config_snapshot_path),
    "artifact.type": "config_snapshot",
})
```

### Cross-SLURM-job context propagation

```python
# In submit.sh or the training entrypoint, after stage N completes:
carrier = {}
propagate.inject(carrier)
run_record["otel_context"] = carrier  # persisted in run_record.json

# In stage N+1, at startup:
upstream_record = read_run_record(upstream_run_dir)
ctx = propagate.extract(upstream_record.get("otel_context", {}))
with tracer.start_as_current_span("stage.supervised", context=ctx):
    ...
```

## 6. Backend Options for HPC

GraphIDS runs on OSC (headless Linux, no always-on services). Backend
selection must work without a persistent collector.

| Backend | Pros | Cons | Verdict |
|---------|------|------|---------|
| **JSONL files** (custom exporter) | Zero infra, works offline, queryable with jq/DuckDB/polars | No real-time dashboard, requires post-hoc analysis | **Primary for HPC** |
| **OTLP to collector** | Standard protocol, feeds Jaeger/Grafana/etc. | Requires running a collector process, not available on OSC login nodes | Secondary (local dev only) |
| **Console** | Zero config, instant feedback | Not structured, not queryable | Dev/debug only |
| **DuckDB direct** (custom exporter) | SQL-queryable immediately, fits existing catalog pattern | Custom exporter needed, concurrent writes from SLURM array jobs need care | Future option — write JSONL, query with `duckdb-otlp` extension or rebuild script |
| **Wandb** (custom exporter) | Existing dashboard, team sharing | Proprietary, requires network, sync daemon overhead | Optional overlay, not primary |

### Recommended architecture for HPC

```
Training job (SLURM)
  └── OTel SDK
       ├── TracerProvider  → SimpleSpanProcessor → JsonlFileSpanExporter  → {run_dir}/traces.jsonl
       ├── MeterProvider   → PeriodicReader      → JsonlFileMetricExporter → {run_dir}/metrics.jsonl
       └── LoggerProvider  → BatchLogProcessor   → JsonlFileLogExporter   → {run_dir}/logs.jsonl

Post-job analysis
  └── DuckDB reads JSONL files (or duckdb-otlp extension reads traces.jsonl)
       └── Catalog rebuild from spans instead of run_record.json
```

SimpleSpanProcessor for spans (immediate write, no data loss on SLURM
preemption or wall-time kill). PeriodicReader for metrics (batched, lower
I/O). BatchLogProcessor for logs (high volume, buffered).

## 7. DuckDB Integration

The `duckdb-otlp` community extension can read OTLP JSONL directly:

```sql
INSTALL otlp FROM community;
LOAD otlp;

-- Query all training spans
SELECT TraceId, SpanName, Duration / 1e6 AS duration_ms,
       attributes->>'ml.epoch' AS epoch,
       attributes->>'ml.train.loss' AS loss
FROM read_otlp_traces('experimentruns/production/hcrl_sa/*/traces.jsonl')
WHERE SpanName LIKE 'training.epoch%'
ORDER BY duration_ms DESC;
```

This could replace the current `rebuild-catalog` command that parses
`run_record.json` sidecars into a DuckDB table. The trace data is richer
(timing, parent-child relationships, events) and the query interface is
the same (SQL).

Alternatively, a `RunRecordSpanExporter` can continue writing
`run_record.json` from span attributes, maintaining backward compatibility
with the existing catalog rebuild.

## 8. Integration with Lightning

Two approaches:

### A. OTel as a Lightning Logger (subclass `lightning.pytorch.loggers.Logger`)

```python
class OTelLogger(Logger):
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self._histograms[name].record(value, {"step": step})
            span = trace.get_current_span()
            span.add_event(f"metric.{name}", {"value": value, "step": step})

    def log_hyperparams(self, params: dict) -> None:
        span = trace.get_current_span()
        for k, v in _flatten(params).items():
            span.set_attribute(f"hparam.{k}", _coerce(v))
```

Pro: Lightning's `self.log()` calls flow through automatically.
Con: Logger interface is limited (no trace lifecycle control).

### B. OTel as a Lightning Callback (the mock's approach)

```python
class OTelCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        self._fit_span = self._tracer.start_span("training.fit")
        # ... set resource attributes

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._loss_histogram.record(outputs["loss"].item())
        self._batch_counter.add(1)
        # Record VRAM
        self._cuda_gauge.set(torch.cuda.memory_allocated() / 1e6)

    def on_fit_end(self, trainer, pl_module):
        self._fit_span.end()
```

Pro: Full control over span lifecycle. Can create spans for epochs, batches,
validation, checkpoint saves.
Con: Must manually read `trainer.callback_metrics`.

**Recommendation: Both.** Use a Logger for `self.log()` integration (metrics
flow through without changing model code). Use a Callback for span lifecycle,
resource profiling, and artifact tracking (things the Logger interface can't do).

## 9. `spawn` Multiprocessing Considerations

GraphIDS uses `spawn` for DataLoader workers (critical constraint: never
`fork` with CUDA). OTel implications:

1. **TracerProvider is not fork-safe** — but with `spawn`, each worker is a
   fresh process. Workers would need their own TracerProvider initialized
   after spawn.

2. **DataLoader workers don't need tracing.** They load/transform data and
   return tensors. The training loop (main process) owns all spans. No
   OTel setup needed in workers.

3. **Cross-process context for pipeline stages** (autoencoder -> supervised)
   uses file-based carrier dicts, not shared memory. Fully compatible
   with spawn.

4. **`BatchSpanProcessor` uses threading** — no conflict with spawn. The
   background export thread lives in the main process.

No special handling needed. This is simpler than the mock anticipated.

## 10. Migration Path

### Phase 1: Foundation (no behavior change)

- Add `graphids/telemetry/` module with provider setup and JSONL exporters.
- Wire into `train_entrypoint.py` alongside existing logging.
- Both systems write simultaneously. Validate JSONL output matches
  existing run_record.json data.

### Phase 2: Replace callbacks

- New `OTelCallback` replaces `ResourceProfileCallback` (fixing the
  missing-rows bug) and `RunRecordCallback`.
- `RunRecordSpanExporter` writes `run_record.json` from span data,
  maintaining DuckDB catalog compatibility.
- Delete `ResourceProfileCallback` and `RunRecordCallback`.

### Phase 3: Replace loggers

- `OTelLogger` replaces `WandbLogger` + `CSVLogger` as primary.
- Optional `WandbExporter` keeps Wandb dashboard if desired.
- `graphids/log.py` simplified to just `get_logger()` + OTel bridge setup
  (delete `_JSONFormatter`, `_SlurmFilter`, `_StructuredAdapter`).

### Phase 4: Cross-stage tracing

- Inject OTel context into `run_record.json` (or a `traceparent` file).
- Extract context at stage startup.
- Pipeline runs become single trace trees queryable in DuckDB.

### Phase 5: Catalog migration

- Evaluate `duckdb-otlp` extension for direct JSONL querying.
- If viable, simplify `rebuild-catalog` to read from traces.jsonl
  instead of parsing run_record.json sidecars.

## 11. Dependency Impact

### New dependencies

```
opentelemetry-api >= 1.25
opentelemetry-sdk >= 1.25
opentelemetry-semantic-conventions >= 0.46b0
```

### Potentially removable

```
wandb  # if fully replaced by OTel + JSONL/DuckDB analysis
```

### PyG compatibility

OTel packages are pure Python — no C extensions, no CUDA coupling, no
version coupling with PyTorch or PyG. Safe to add without affecting the
PyTorch/PyG/CUDA constraint triangle.

## 12. Open Questions

1. **Wandb retention.** Do we still want the Wandb UI for interactive
   dashboards, or is DuckDB + notebook analysis sufficient? If yes, write
   a `WandbExporter` that translates spans/metrics to `wandb.log()` calls.

2. **Metric export interval.** `PeriodicExportingMetricReader` defaults
   to 60s. For training loops with ~100 batches/epoch on small datasets,
   this may be too coarse. Tunable per run.

3. **JSONL file size.** Per-batch span events could generate large
   traces.jsonl files for long training runs. Options: sample batches,
   only record epoch-level spans, or compress with gzip.

4. **DuckDB `otlp` extension maturity.** Community extension — verify it
   handles the JSONL format our custom exporter produces, or stick with
   the existing catalog rebuild from run_record.json.

5. **Observable Gauge vs UpDownCounter.** OTel SDK 1.25+ added
   synchronous `Gauge` instrument. If we target 1.25+, we don't need the
   mock's UpDownCounter delta-tracking workaround.

6. **SLURM preemption.** On `SIGUSR1` (5 min before wall time),
   `SimpleSpanProcessor` has already flushed all completed spans. But
   the in-progress epoch span would be lost. Options: flush on signal
   handler, or accept partial trace on preemption.

## 13. Recommendation

**Adopt OTel as the unified observability layer.** The mock implementation
demonstrates the design is sound. Key advantages:

- **Single identity** across all signals (Resource attributes).
- **Automatic log-trace correlation** via the logging bridge.
- **Cross-stage tracing** via context propagation — the pipeline becomes
  one queryable trace tree.
- **Fixes the ResourceProfileCallback bug** by design — device stats
  become OTel gauge instruments recorded in `on_train_batch_end`.
- **Vendor neutral.** JSONL files for HPC, OTLP for local dev with
  Jaeger/Grafana, Wandb exporter for team dashboards — all from the
  same instrumentation code.
- **Pure Python deps.** No CUDA/PyG coupling risk.
- **Net code reduction.** Replaces ~300 lines of custom logging/callback/
  sidecar code with ~200 lines of OTel setup + standard SDK calls.

The phased migration path allows incremental adoption with no flag day.
Start with Phase 1 (dual-write) to validate, then cut over.
