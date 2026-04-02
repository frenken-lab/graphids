# Ablation Bug Patterns — Prevention Guide

> Extracted from backlog 2026-04-02. Bugs 1-10 all fixed.

**Pattern A — Stale tests after code changes (bugs 1-3):**
Tests hardcoded old values from before sessions 6-7. Code was correct, tests drifted.
Prevention: run `scripts/submit.sh tests` after any code change before launching runs.

**Pattern B — Stale artifacts from prior code structure (bugs 5-6):**
Config YAML copied from another model without removing model-specific fields.
Old checkpoints with class_paths from pre-reorganization module layout.
Prevention: `orchestrate validate` catches YAML parse errors. Clear stale checkpoints
when module paths change.

**Pattern C — Identity keys ≠ model keys (bugs 7-8):**
Identity keys (for hash dedup) were blindly passed as `--model.init_args.X` CLI overrides.
Not all identity keys are valid model init params. Prevention: `model_keys` (subset of
`identity_keys`) controls which keys become CLI overrides.
