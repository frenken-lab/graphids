# Phase E: Scripts, Docs, and Stale Reference Cleanup

**Date:** 2026-03-20
**Status:** Ready
**Depends on:** Phase C (done), Phase D (done)

---

## Problem

Phase C deleted `graphids/storage/` but left behind:
1. **3 scripts** that import from `graphids.storage` (broken at runtime)
2. **1 shell script** with inline Python importing `graphids.storage.manifest` (broken)
3. **3 rules files** describing a storage layer that no longer exists
4. **1 CLAUDE.md** with stale architecture references
5. **Stale docstrings** referencing StorageGateway, `_manifest.json`, `compose_config()`
6. **`ray` optional dep** in pyproject.toml (nothing uses it)
7. **`_protocols.py`** StageMetrics docstring references `_manifest.json`
8. **`api.py`** docstring references "manifest"

---

## Change inventory

### Group 1: Fix broken scripts (3 Python files + 1 shell script)

These scripts import from `graphids.storage.paths` which no longer exists. The functions they need (`lake_root_from_env`, `lake_catalog_path`) were moved to `graphids.config.paths` in Phase C. Two functions (`lake_exports_dir`, `lake_run_dir`) were NOT moved — they need to be added to `config/paths.py` or inlined.

**1a. `scripts/data/push_experiments_to_hf.py`** (lines 27-28)

Current:
```python
    from graphids.storage.paths import lake_catalog_path, lake_root_from_env
    from graphids.storage.catalog import rebuild_catalog
```

`lake_catalog_path` and `lake_root_from_env` exist in `graphids.config.paths`. `rebuild_catalog` was in `graphids.storage.catalog` which is deleted. The whole script's data flow (`_manifest.json → rebuild_catalog → DuckDB → parquet`) is broken because there are no `_manifest.json` files being written anymore (Phase C removed manifest writes).

**Decision:** Rewrite to read from `metrics.csv` + `hparams.yaml` (CSVLogger output) instead of manifests. This is the replacement the research plan specified.

New approach:
```python
    from graphids.config.paths import lake_root_from_env

    lake_root = lake_root_from_env()
    if lake_root is None:
        log.error("lake_root_not_set")
        return

    # Glob for metrics.csv files written by CSVLogger
    import pandas as pd
    from pathlib import Path

    csv_files = list(Path(lake_root).rglob("metrics.csv"))
    if not csv_files:
        log.warning("no_metrics_files_found")
        return

    frames = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Extract identity from path: .../dataset/model_scale_stage/seed_N/metrics.csv
        parts = f.relative_to(lake_root).parts
        if len(parts) >= 4:
            df["run_dir"] = str(f.parent)
            # Try to read hparams.yaml for config identity
            hparams = f.parent / "hparams.yaml"
            if hparams.exists():
                import yaml
                hp = yaml.safe_load(hparams.read_text())
                if isinstance(hp, dict) and "cfg" in hp:
                    cfg = hp["cfg"]
                    df["dataset"] = cfg.get("dataset", "")
                    df["model_type"] = cfg.get("model_type", "")
                    df["scale"] = cfg.get("scale", "")
                    df["seed"] = cfg.get("seed", "")
        frames.append(df)

    runs = pd.concat(frames, ignore_index=True)
    log.info("runs_found", count=len(runs))
```

Full replacement file (~55 lines, down from 79).

**1b. `scripts/data/export_paper_data.py`** (line 28)

Current:
```python
from graphids.storage.paths import lake_exports_dir, lake_root_from_env, lake_run_dir
```

Needs: `lake_root_from_env` (exists in config.paths), `lake_exports_dir` (trivial: `Path(root) / "exports"`), `lake_run_dir` (no longer exists — was `{root}/{tier}/{dataset}/{model}_{scale}_{stage}/seed_{N}`).

**Decision:** Add `lake_exports_dir` to `graphids/config/paths.py` (2 lines). Inline `lake_run_dir` as a local helper in the script (the Hydra `run.dir` template is the canonical path now, but this script needs to find existing run dirs on disk).

Edit line 28:
```python
from graphids.config.paths import lake_root_from_env
```

Add local helpers after imports:
```python
def lake_exports_dir(lake_root):
    return Path(lake_root) / "exports"

def lake_run_dir(lake_root, dataset, model_type, scale, stage, seed=42, production=False):
    tier = "production" if production else f"dev/{os.environ.get('USER', 'unknown')}"
    return Path(lake_root) / tier / dataset / f"{model_type}_{scale}_{stage}" / f"seed_{seed}"
```

**1c. `scripts/data/generate_attack_type_mapping.py`** (line 26)

Current:
```python
from graphids.storage.paths import lake_exports_dir, lake_root_from_env
```

**Decision:** Same pattern — use config.paths + local `lake_exports_dir` helper.

Edit line 26:
```python
from graphids.config.paths import lake_root_from_env
```

Add local helper:
```python
def lake_exports_dir(lake_root):
    return Path(lake_root) / "exports"
```

**1d. `scripts/lake/migrate_to_ess.sh`** (line 130)

Current (inline Python):
```python
from graphids.storage.manifest import write_manifest, read_manifest
```

This entire inline Python block generates `_manifest.json` for migrated runs. Since we no longer write manifests, this block is dead code.

**Decision:** Delete the manifest generation block (lines 126-160ish). The migration script's rsync portion still works; only the manifest generation is broken.

### Group 2: Add `lake_exports_dir` to config/paths.py

Rather than duplicating the helper in each script, add it once to `graphids/config/paths.py`:

```python
def lake_exports_dir(lake_root: str | Path) -> Path:
    """Path: {lake_root}/exports"""
    return Path(lake_root) / "exports"
```

Then scripts 1b and 1c import from `graphids.config.paths` instead of using local helpers.

Also add to `graphids/config/__init__.py` re-exports.

### Group 3: Remove `ray` optional dep from pyproject.toml

Current (line 42):
```toml
ray = ["ray[default]>=2.49", "ray[tune]>=2.49"]
```

And line 95:
```toml
    "ray.*",
```

Ray is only used in `core/preprocessing/_parallel.py` as a soft optional (`_ray_available()` checks `import ray`). The optional dep group can be removed — anyone who wants Ray parallelism can install it manually. The code gracefully falls back to multiprocessing.

**Decision:** Delete the `ray` optional dep group from pyproject.toml. Leave the runtime soft-import in `_parallel.py` (it still works if ray happens to be installed).

### Group 4: Update rules files (3 files)

**4a. `.claude/rules/architecture.md`**

Major rewrite needed. Current file describes:
- Storage layer (lines 29-55) — deleted
- Dagster orchestration (lines 57-81) — replaced by submitit + graphlib
- Dagster Pipes client, slurm_primitives — deleted
- References `optuna_sweep.py` — deleted
- References `cli.py` — replaced by `__main__.py`
- References `ArtifactMapper` — deleted
- References `compose_config()` — deleted
- References `checkpoint_path` — deleted

Edits:
- Delete "Storage Layer" section entirely (lines 29-55)
- Update "Orchestration" section: remove Dagster references, describe submitit + graphlib DAG
- Update "Evaluation" section: remove ArtifactMapper references, mention EvalArtifactCallback
- Remove HPO Optuna reference (optuna_sweep.py deleted)
- Update CLI reference from `cli.py` to `__main__.py`
- Update "Shared Principles" — remove archive restore, subprocess dispatch for sweep

**4b. `.claude/rules/code-style.md`**

Current lines 7-19 describe a 4-layer import hierarchy with `graphids/storage/` as Layer 0. Now it's 3 layers.

Replace lines 5-19:
```markdown
## Import Rules (3-layer hierarchy)

Enforced by `tests/test_layer_boundaries.py`:

1. **`graphids/config/`** (top): Never imports from `pipeline/` or `core/`.
2. **`graphids/pipeline/`** (middle): Imports `graphids.config` freely at top level. Imports `graphids.core` only inside functions (lazy).
3. **`graphids/core/`** (bottom): Imports `graphids.config.constants` for shared constants. Never imports from `graphids.pipeline`.

When adding new code:
- Constants → `graphids/config/constants.py`
- Hyperparameters → Pydantic models in `graphids/config/schema.py`
- Architecture defaults → YAML files in `graphids/config/conf/model/` or `graphids/config/conf/auxiliary/`
- Path helpers → `graphids/config/paths.py`
- `from graphids.config import PipelineConfig, resolve` — use the package re-exports
```

**4c. `.claude/rules/project-structure.md`**

Full rewrite of the tree. Major changes:
- Remove `storage/` entirely (lines 10-17)
- Remove `lake/` (lines 37-40)
- Remove `search_spaces/` (line 35)
- Remove `cli.py` (line 43), `subprocess_utils.py` (line 45)
- Remove `dagster_defs.py`, `pipes_slurm.py`, `slurm_primitives.py`, `optuna_sweep.py` (lines 61-64)
- Add `__main__.py`
- Add `callbacks.py` to stages/
- Add `cka.py` to stages/
- Update `trainer_factory.py` description (no longer mentions ModelCheckpoint/EarlyStopping directly)
- Update orchestration/ to show `dag.py`, `job.py`, `slurm.py` (3 files, not 5)
- Update `__init__.py` descriptions
- Update file count

### Group 5: Stale docstrings and comments (minor)

**5a. `graphids/core/models/_protocols.py:26`**
```python
    """Contract: what every stage returns in the metrics dict (written to _manifest.json)."""
```
Change to:
```python
    """Contract: what every stage returns in the metrics dict."""
```

**5b. `graphids/api.py:32`**
```python
    """Train a model. Returns StageResult with metrics, checkpoint path, manifest."""
```
Change to:
```python
    """Train a model. Returns StageResult with metrics and checkpoint path."""
```

**5c. `graphids/api.py:8`**
```python
    # Train a single stage (full guarantees: validation, manifest, logging)
```
Change to:
```python
    # Train a single stage (full guarantees: validation, logging)
```

**5d. `graphids/pipeline/stages/evaluation.py:37`**
```python
    cli.py passes this to the manifest (single source of truth for metrics).
```
Change to:
```python
    Metrics are logged via CSVLogger (single source of truth).
```

**5e. `graphids/pipeline/stages/trainer_factory.py:195-196`**
```python
    """Load the frozen config.json saved during training for *stage*.

    model_type defaults to the canonical owner of the stage (e.g. "autoencoder" → "vgae").
    Uses the StorageGateway for filesystem resolution.
```
Remove the StorageGateway line.

**5f. `graphids/pipeline/stages/trainer_factory.py:222-223`**
```python
    """Load a trained model using its frozen config and the registry.

    Uses the StorageGateway for filesystem resolution.
```
Remove the StorageGateway line.

### Group 6: Update PLAN.md

- Mark Phase C as **Done** (currently says "next")
- Mark Phase D as **Done**
- Mark Phase E as **Done** (after this is committed)
- Remove stale I/O pillar from 3-Pillar Architecture table
- Update file counts / net lines

---

## Execution order

1. Add `lake_exports_dir` to `graphids/config/paths.py` and `__init__.py`
2. Fix 3 broken scripts (update imports)
3. Fix `migrate_to_ess.sh` (delete manifest generation block)
4. Remove `ray` optional dep from pyproject.toml
5. Update `.claude/rules/architecture.md`
6. Update `.claude/rules/code-style.md`
7. Update `.claude/rules/project-structure.md`
8. Fix stale docstrings (5 files, 6 edits)
9. Update PLAN.md

---

## Files changed (summary)

| File | Action | Net lines |
|------|--------|---:|
| `graphids/config/paths.py` | Add `lake_exports_dir` | +4 |
| `graphids/config/__init__.py` | Add re-export | +1 |
| `scripts/data/push_experiments_to_hf.py` | Rewrite: manifests → metrics.csv | ~-20 |
| `scripts/data/export_paper_data.py` | Fix import + add local helper | ~+5 |
| `scripts/data/generate_attack_type_mapping.py` | Fix import | ~0 |
| `scripts/lake/migrate_to_ess.sh` | Delete manifest gen block | ~-35 |
| `pyproject.toml` | Remove ray optional dep | -2 |
| `.claude/rules/architecture.md` | Rewrite (remove storage, update orchestration) | ~-40 |
| `.claude/rules/code-style.md` | 4-layer → 3-layer | ~-8 |
| `.claude/rules/project-structure.md` | Full tree rewrite | ~-20 |
| `_protocols.py`, `api.py`, `evaluation.py`, `trainer_factory.py` | Fix stale docstrings | ~-6 |
| `PLAN.md` | Update phase statuses | ~+5 |
| **Total** | | **~-120** |

---

## Verification

- `grep -r "graphids\.storage" graphids/ scripts/ .claude/rules/` → zero matches
- `grep -r "_manifest\.json" graphids/` → zero matches (only in preprocessing `feature_manifest.json` which is different)
- `python -c "from graphids.config.paths import lake_exports_dir"` → no error
- Scripts are not tested on login node (they need KD_GAT_LAKE_ROOT + HF_TOKEN), but imports should resolve
