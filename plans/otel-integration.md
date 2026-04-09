# OTel Integration — Completed

> Implemented: 2026-04-08 (session 39) | Architecture: `docs/reference/observability.md`

## SDK Gotchas (reference for future OTel work)

| Gotcha | Detail |
|--------|--------|
| `get_tracer_provider()` returns API type | No `add_span_processor` method. Keep SDK `TracerProvider` reference for adding processors after init. |
| `MeterProvider` readers are constructor-only | Cannot add readers after creation. Phase B creates a new `MeterProvider` and replaces the global. |
| `otel-file-exporter` requires Python 3.13+ | Unusable on OSC (3.12). Use `ConsoleSpanExporter(out=file)` / `ConsoleMetricExporter(out=file)` instead. |
| `create_histogram()` must be cached | Create once per metric name, reuse. `OTelTrainingLogger._instruments` dict handles this. |
| Wandb Weave OTLP endpoint | `https://trace.wandb.ai/otel/v1/traces`, auth via `wandb-api-key` header. Login node verified; compute node unverified. |
| `BatchSpanProcessor` silently drops on timeout | Training won't hang if Weave is unreachable from compute nodes. |
| SIGUSR1 flush | `atexit.register` handles normal exit. SLURM preemption needs explicit `_tp.shutdown()` in signal handler. |
