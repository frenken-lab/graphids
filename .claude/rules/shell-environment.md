---
paths:
  - "scripts/**"
  - "*.sh"
  - "pyproject.toml"
---

# KD-GAT Shell Environment (CANONICAL)

**Updated 2026-02-24.** This is the canonical shell environment for KD-GAT. Uses uv + `.venv/`, NOT conda.

## Setup

```bash
# Activate the project venv (REQUIRED before any Python commands):
source ~/KD-GAT/.venv/bin/activate

# Or set paths explicitly:
export PATH="$HOME/KD-GAT/.venv/bin:$PATH"
export PYTHONPATH=/users/PAS2022/rf15/KD-GAT:$PYTHONPATH

# Then run:
python -m graphids.cli ...
python -m pytest tests/ -v
```

## Package Manager

- **uv 0.10+** at `~/.local/bin/uv`
- Venv created with: `uv venv --python /apps/python/3.12/bin/python3`
- Install/sync: `uv sync --extra dev`
- Lock file: `uv.lock` (committed)

## Common Failures

- `ModuleNotFoundError: No module named 'graphids'` → missing PYTHONPATH or venv not activated
- `ModuleNotFoundError: No module named 'torch'` → venv not activated (using system Python)
- Never use conda — the project uses uv + `.venv/` exclusively
