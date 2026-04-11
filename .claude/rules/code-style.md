# GraphIDS Code Style

## Logging Style

- `from graphids._otel import get_logger; log = get_logger(__name__)` — stdlib logging with structured kwargs adapter
- Structured events: `log.info("event_name", key=value)` — no format strings
- Handlers (OTel LoggingHandler, SLURM sinks) are installed by `init_providers()` in `graphids/_otel.py`, called from the Typer root callback in `graphids/cli/app.py`. Level is set there via `--verbose/-v` on the root command.
- Under SLURM: JSONL goes to `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`. Otherwise: human-readable stderr.

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
