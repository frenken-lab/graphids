# GraphIDS Code Style

## Import Rules (3-layer hierarchy)

1. **`graphids/config/`** (top): Never imports from `pipeline/` or `core/`.
2. **`graphids/pipeline/`** (middle): Imports `graphids.config` freely at top level. Imports `graphids.core` only inside functions (lazy).
3. **`graphids/core/`** (bottom): Imports `graphids.config.constants` for shared constants. Never imports from `graphids.pipeline`.

When adding new code:
- Constants → `graphids/config/constants.py`
- Schema (dataclasses) → `graphids/config/__init__.py`
- `from graphids.config import resolve, Config` — use the package re-exports

## Logging Style

- `from graphids.log import get_logger; log = get_logger(__name__)` — stdlib logging with structured kwargs adapter
- Structured events: `log.info("event_name", key=value)` — no format strings
- Configuration via `graphids.log.configure_logging()` in `__main__.py` and `definitions.py`
- Under SLURM: JSONL to `{SLURM_LOG_DIR}/orchestrator_{job_id}.jsonl`. Otherwise: human-readable stderr.

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
