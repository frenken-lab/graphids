# KD-GAT Code Style

## Import Rules (4-layer hierarchy)

Enforced by `tests/test_layer_boundaries.py`:

0. **`graphids/storage/`** (infrastructure): No imports from `config/`, `pipeline/`, or `core/`. `gateway.py` and `paths.py` are fully domain-free. `mapper.py` uses lazy (function-local) domain imports only.
1. **`graphids/config/`** (top): Imports `graphids.storage` for path primitives. Never imports from `pipeline/` or `core/`.
2. **`graphids/pipeline/`** (middle): Imports `graphids.config` and `graphids.storage` freely at top level. Imports `graphids.core` only inside functions (lazy).
3. **`graphids/core/`** (bottom): Imports `graphids.config.constants` for shared constants. Uses `graphids.storage` for cache I/O (lazy). Never imports from `graphids.pipeline`.

When adding new code:
- Constants → `graphids/config/constants.py`
- Hyperparameters → Pydantic models in `graphids/config/schema.py`
- Architecture defaults → YAML files in `graphids/config/conf/model/` or `graphids/config/conf/auxiliary/`
- Path helpers → `graphids/config/paths.py` (PipelineConfig-based) or `graphids/storage/paths.py` (raw lake layout)
- File I/O → `graphids/storage/gateway.py` (transport) or `graphids/storage/mapper.py` (domain-aware serialization)
- `from graphids.storage import open_gateway` — standard gateway+mapper creation
- `from graphids.config import PipelineConfig, resolve, checkpoint_path` — use the package re-exports

## General Style

- Make minimal, targeted changes. Prefer the simplest solution; avoid speculative abstractions.
- Leave untouched code untouched — only add docstrings, comments, or type annotations to lines you changed.
- If something is unused, delete it completely. No compatibility shims.

## Iteration Hygiene

Before implementing a new feature or fix:
1. **Audit touchpoints** — identify files that will be modified
2. **Cut stale code** — remove dead code in those files
3. **Simplify** — replace complex patterns with simpler ones
4. **Delete, don't comment** — unused code gets deleted

## Repeated Failure Protocol

If the same command fails twice for the same category of reason, STOP retrying and diagnose:
1. Name the pattern (env setup, path resolution, config incompatibility, NFS issue)
2. Read the full traceback — root cause is usually in the first or last frame
3. Check known issues in `critical-constraints.md` and `knowledge-bank.md`
4. Fix the root cause, not the symptom
5. Document if it will recur

## Git

- Short summary line, body explains why not what.
- Push via SSH (`git@github.com:`), not HTTPS.
