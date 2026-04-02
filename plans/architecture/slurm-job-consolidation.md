# Consolidate SLURM Jobs: train → test → analyze in one job

> **Supersedes**: `plans/architecture/evaluation-analysis-assets.md` (separate dagster assets per phase). That design was written before we discovered that more dagster assets = more failure surface (env var propagation, multiprocess executor child failures, in-process torch on CPU). This plan bundles phases inside the SLURM job and lets dagster handle only inter-model dependencies.

## Context

Each model config currently produces 2 dagster assets (training + analysis) and 2 SLURM jobs worth of work. The analysis asset runs **in-process** on the dagster CPU node — importing torch without GPU. This is broken (issue: `analysis-assets-in-process.md`). Evaluation (test set metrics) doesn't exist at all (issue: `evaluation-stage-missing.md`).

**Goal**: Each SLURM GPU job runs 3 sequential CLI commands. Each model config is one dagster asset, one SLURM job.

```bash
#!/bin/bash
source scripts/slurm/_preamble.sh
python -m graphids train-from-spec --spec-file $SPEC_FILE
python -m graphids test-from-spec  --spec-file $SPEC_FILE     # NEW
python -m graphids analyze-from-spec --spec-file $SPEC_FILE   # MOVED from in-process
source scripts/slurm/_epilog.sh
```

`set -e` in preamble ensures: if train fails, test and analyze don't run.

## Bugs Addressed

| Bug | Issue | How this plan fixes it |
|-----|-------|----------------------|
| Analysis runs in-process on CPU | `issues/analysis-assets-in-process.md` | **Fixed** — analyze runs inside GPU SLURM job, not in dagster worker |
| No evaluation stage | `issues/evaluation-stage-missing.md` | **Fixed** — `test-from-spec` runs LightningCLI `test` in same job |
| Multiprocess executor child failures | 5 `kd-gat-ablation` jobs failed (session 2026-04-01) | **Reduced** — fewer assets = fewer executor workers to fail |
| Recipe env var lost across processes | `issues/recipe-env-var-not-propagating.md` | **Reduced** — fewer process boundaries. Code fix (move to `build_defs`) also applied |

## Bugs NOT Addressed (need separate fixes)

| Bug | Issue | Why not addressed |
|-----|-------|------------------|
| `runtime.py` module-level env vars (11 constants) | Same class as recipe bug | Architecture change doesn't fix import-time reads. **Prerequisite fix**: move all `os.environ.get()` in `runtime.py` to lazy reads |
| Pipeline observability on headless OSC | `issues/pipeline-observability.md` | Orthogonal — needs CLI status command regardless of SLURM job structure |
| `open_issues.md` deferred items | `plans/open_issues.md` | Unrelated (dead code, DataModule issues, scratch cleanup) |

## Prerequisite: Fix `runtime.py` module-level env vars

Before implementing this plan, move all 11 `os.environ.get()` calls in `graphids/config/runtime.py` from module scope to lazy reads (functions or properties). These are the same class of bug as the recipe env var issue — frozen at import time, silently fall back to defaults if env var isn't set yet.

Current (fragile):
```python
LAKE_ROOT: str = os.environ.get("KD_GAT_LAKE_ROOT", ...)  # module level
```

Target (robust):
```python
def lake_root() -> str:
    return os.environ.get("KD_GAT_LAKE_ROOT", ...)
```

All consumers (`from graphids.config import LAKE_ROOT`) become function calls. This is a breaking API change — every import site needs updating.

## Design Decisions

**Single spec file, not three.** The training spec already has model_family, dataset, run_dir, config_files, upstream_ckpt_paths. Test and analyze can derive what they need:
- `test-from-spec`: Same config files + checkpoint from `{run_dir}/checkpoints/best_model.ckpt` → LightningCLI `test`
- `analyze-from-spec`: Already accepts a spec file. Extend the training spec with optional analysis fields, or write a second spec file alongside the training spec.

**Option A (chosen): Write two spec files.** `run_training_job()` already writes the training spec. Add a second write for the analysis spec alongside it. This keeps the existing contract boundary clean — TrainingSpec for train/test, AnalysisSpec for analyze. The SLURM script references both files.

**Skip analysis/test via recipe config.** Recipes can set `evaluate: false` to skip test, and analysis is auto-skipped for unsupported model types (fusion, temporal).

## Implementation Steps

### Step 1: Add `test-from-spec` command

**New file: `graphids/commands/test_from_spec.py`**

Pattern: identical to `train_from_spec.py`. Loads spec, builds CLI args with `test` instead of `fit`, adds `--ckpt_path` pointing to best checkpoint.

```python
def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--spec-file", required=True)
    args = p.parse_args(argv)
    payload = load_payload(args.spec_file)
    spec = TrainingContract.from_envelope(payload)
    run_test_from_spec(spec)
```

**New function in `graphids/core/train_entrypoint.py`:**

```python
def run_test_from_spec(spec: TrainingSpec) -> None:
    ckpt = Path(spec.run_dir) / "checkpoints" / "best_model.ckpt"
    if not ckpt.exists():
        log.warning("no_checkpoint_for_test", run_dir=spec.run_dir)
        return
    args = ["test"]
    for cf in spec.config_files:
        args.extend(["--config", cf])
    overrides = TrainingContract.to_override_dict(spec)
    for key, val in overrides.items():
        args.append(f"--{key}={val}")
    args.append(f"--ckpt_path={ckpt}")
    run_lightning(args)
```

**Register in `graphids/__main__.py`:**
Add `"test-from-spec": "graphids.commands.test_from_spec"` to `_COMMAND_MODULES`.

### Step 2: Modify script generation to include all 3 commands

**File: `graphids/slurm/slurm.py` — `generate_script()`**

Change to accept phase flags and optional analysis spec path:

```python
def generate_script(resources, *, spec_file, run_test=True,
                    analysis_spec_file=None):
    quoted = shlex.quote(str(spec_file))
    lines = [
        "#!/bin/bash",
        f"source {PROJECT_ROOT}/scripts/slurm/_preamble.sh",
        f"python -m graphids train-from-spec --spec-file {quoted}",
    ]
    if run_test:
        lines.append(f"python -m graphids test-from-spec --spec-file {quoted}")
    if analysis_spec_file:
        aquoted = shlex.quote(str(analysis_spec_file))
        lines.append(f"python -m graphids analyze-from-spec --spec-file {aquoted}")
    lines.append(f"source {PROJECT_ROOT}/scripts/slurm/_epilog.sh")
    return "\n".join(lines) + "\n"
```

### Step 3: Wire analysis spec into the SLURM job

**File: `graphids/slurm/slurm.py` — `run_training_job()`**

Add `analysis_spec: AnalysisSpec | None = None` and `run_test: bool = True` parameters. If analysis_spec provided, write it alongside the training spec and pass its path to `generate_script()`.

**New function: `write_analysis_spec()`** — mirrors `write_training_spec()`, writes `AnalysisContract.to_envelope()` JSON.

### Step 4: Merge training + analysis into single dagster asset

**File: `graphids/orchestrate/assets.py`**

Delete `make_analysis_asset()` entirely. Modify `make_training_asset()`:

- Import `build_analysis_spec`, `supports_analysis` from `orchestrate.analysis`
- After `resolver.resolve()`, build analysis spec if model supports it
- Pass `analysis_spec` and `run_test` to `submit_and_wait()`
- Return checkpoint path (unchanged — downstream assets only need the checkpoint)

### Step 5: Simplify `build_defs()`

**File: `graphids/orchestrate/component.py` — `build_defs()`**

Remove:
- `analysis_sources` filtering (line 103)
- `analysis_assets` list (line 104)
- `assets.extend(analysis_assets)` (line 105)
- `make_analysis_checks()` call (line 111)

### Step 6: Merge checks

**File: `graphids/orchestrate/checks.py`**

Merge `make_checkpoint_checks()` and `make_analysis_checks()` into a single `make_asset_checks()`. The check verifies:
1. Checkpoint file exists
2. `.complete` marker exists
3. If model supports analysis: analysis manifest exists and all expected outputs present

## Files Modified

| File | Change |
|------|--------|
| `graphids/commands/test_from_spec.py` | **NEW** — test-from-spec command |
| `graphids/__main__.py:40-51` | Register `test-from-spec` |
| `graphids/core/train_entrypoint.py` | Add `run_test_from_spec()` |
| `graphids/slurm/slurm.py:65-81` | `generate_script()` → multi-command |
| `graphids/slurm/slurm.py:179-203` | `run_training_job()` → accept analysis_spec, run_test |
| `graphids/slurm/slurm.py` | Add `write_analysis_spec()` |
| `graphids/orchestrate/assets.py:30-103` | Build analysis spec in `_train`, pass to submit |
| `graphids/orchestrate/assets.py:106-151` | **DELETE** `make_analysis_asset()` |
| `graphids/orchestrate/component.py:83-132` | Remove analysis asset + check assembly |
| `graphids/orchestrate/checks.py` | Merge into single `make_asset_checks()` |

## What does NOT change

- `TrainingSpec`, `AnalysisSpec` — existing contracts unchanged
- `train-from-spec`, `analyze-from-spec` commands — work as-is
- `ConfigResolver` — unchanged
- `enumerate_assets()` / `StageConfig` — DAG topology unchanged
- `_preamble.sh`, `_epilog.sh` — unchanged
- IOManager checkpoint handoff between stages — unchanged

## Verification

1. **Unit test**: `test_generate_script_multi_command` — verify script contains all 3 commands
2. **Unit test**: `test_generate_script_skip_analyze` — verify analyze omitted for unsupported models
3. **Collect test**: `python -m pytest --collect-only` — verify test-from-spec command is importable
4. **Smoke test on SLURM**: `scripts/submit.sh ablation --assets '*' --partition 'hcrl_sa|42'` with smoke_test recipe — verify all 3 phases run in one job
5. **Check SLURM logs**: Single job log should show train → test → analyze sequentially
