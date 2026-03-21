# KD-GAT Code Style

## Import Rules (3-layer hierarchy)

1. **`graphids/config/`** (top): Never imports from `pipeline/` or `core/`.
2. **`graphids/pipeline/`** (middle): Imports `graphids.config` freely at top level. Imports `graphids.core` only inside functions (lazy).
3. **`graphids/core/`** (bottom): Imports `graphids.config.constants` for shared constants. Never imports from `graphids.pipeline`.

When adding new code:
- Constants → `graphids/config/constants.py`
- Schema (dataclasses) → `graphids/config/__init__.py`
- `from graphids.config import resolve, Config` — use the package re-exports

## Logging Style

- `import structlog; log = structlog.get_logger()` — never `import logging`
- Structured events: `log.info("event_name", key=value)` — no format strings
- Context via `structlog.contextvars.bind_contextvars()` at entry points
- Logging setup is inlined in `__main__.py`, not a separate module

## General Style

- If something is unused, delete it completely. No compatibility shims.
- If a dependency does it, use the dependency. Don't wrap.
- If it can be inlined, inline it. Don't add single-caller functions.

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
