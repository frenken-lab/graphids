# KD pipeline untested end-to-end

> Created: 2026-04-01 (session 7) | Status: Open

## Problem

The Knowledge Distillation pipeline has never been exercised by a real training run.
The wiring exists (recipe → planning → resolver → CLI → model → teacher loading) but
no recipe contains a `kd:` block, so every link in the chain is untested in production.

## Specific gaps

### 1. No KD recipe exists

`ablation.yaml` has zero `kd:` blocks. The experiment sweep plan (configs #10, #11)
describes KD ablation entries but they were never translated into recipe YAML.

### 2. `prepare_kd` → `checkpoint_path()` never called by pipeline

`prepare_kd` (`_training.py:218`) calls `checkpoint_path()` to locate the teacher
(large-scale) checkpoint. This function was refactored in session 7 to delegate to
`PathContext`. The refactored path produces identical output (verified with unit
test), but has never been exercised by a real SLURM job.

### 3. `prepare_kd` reads hparams that come from LINK_TARGETS

`prepare_kd` accesses `cfg.lake_root` and `cfg.dataset` from model hparams. These
are populated by LightningCLI's `link_arguments` (from `data.init_args.dataset` →
`model.init_args.dataset`). The pipeline path goes through LightningCLI so this
should work, but it's never been tested. If `LINK_TARGETS` ordering or the snapshot
fix (session 7) introduced a regression, it would only surface in KD runs.

### 4. Teacher upstream dependency resolution untested

`planning.py:138-151` scans the recipe for a matching large-scale config to wire as
an upstream dagster dependency. If no large config exists, the teacher checkpoint
must already exist on disk. This scanning logic has only been tested with unit tests
against mock recipe data — never with a real dagster materialization.

### 5. `_KDSpec` / `KDAuxiliary` field divergence

`_KDSpec` (recipe_expand.py) has 7 fields; `KDAuxiliary` TypedDict (_training.py)
has 3. The 4 extra fields (`temperature`, `model_path`, `vgae_latent_weight`,
`vgae_recon_weight`) pass through `runtime_overrides` as a JSON blob without
TypedDict validation. A typo in these fields is only caught at model `_build()` time
inside the SLURM job. (Cross-ref: `docs/backlog/config-overhaul-remaining.md` item P2.4.)

## Options to close

**A. Add KD to smoke test** — include large VGAE + large GAT + KD small sweep.
Adds ~3 jobs and ~1.5h wall time. Validates the full chain.

**B. Add KD to ablation recipe** — where it was always intended (configs #10, #11
from the experiment plan). First real test happens at ablation launch.

**C. Minimal KD wiring test** — hardcode `model_path` to a known teacher checkpoint
in a smoke sweep entry. Bypasses `checkpoint_path()` resolution but tests the
training-side KD wiring (auxiliary loss, teacher loading, projection layer).

Recommendation: **C first** (low cost, tests model-side), then **B** (tests full
chain at ablation time). A is only needed if B fails.
