# GraphIDS Code Style

## Logging Style

- `from graphids.log import get_logger; log = get_logger(__name__)` — stdlib logging with structured kwargs adapter
- Structured events: `log.info("event_name", key=value)` — no format strings
- Configuration via `graphids.log.configure_logging()` in `__main__.py` and `definitions.py`
- Under SLURM: JSONL to `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`. Otherwise: human-readable stderr.

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
