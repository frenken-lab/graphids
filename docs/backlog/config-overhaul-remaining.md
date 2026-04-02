# Config System Overhaul

> Canonical design: `docs/decisions/0007-config-system-architecture.md`
> Created: 2026-03-31 | Audited: 2026-04-02

---

## Completed

Phase 1 (YAML restructuring + forced callbacks), Phase 2.1 (TrainingRunConfig schema),
Phase 2.2 (ConfigResolver with YAML-aware validation), Phase 2.3 (PathContext), W1, W2,
W4 — all done. See audit log in git history for details.

---

## Open — Ordered by Priority

### 1. SLURM validation (blocks confidence)

Run tests and smoke test on SLURM to validate all recent changes.

```bash
scripts/submit.sh tests -k test_overrides
scripts/submit.sh tests -k test_config
```

### 2. W3: Per-stage override granularity (MEDIUM)

`trainer_overrides` and `resource_overrides` apply uniformly to all stages.
"autoencoder gets 2 epochs, curriculum gets 5" is not expressible.
See `docs/backlog/per-stage-overrides.md` for design options.

### 3. P2.4: Align `_KDSpec` / `KDEntry` field sets (LOW-MEDIUM)

`_KDSpec` (recipe_expand.py) has 7 fields; `KDEntry` (contracts.py) has 3. The 4
extra fields bypass `TrainingRunConfig.auxiliaries` validation.
Decision needed: identity-relevant (add to `KDEntry`) or sweep-internal (document split).

### 4. W5: `OverrideRecord.value` type is lossy (LOW)

KD overrides audit stores JSON blob as string. Not queryable.

### 5. W7: Spec-file path bypasses validators (LOW)

Cross-field validation only runs in `ConfigResolver` (dagster side).
`train-from-spec` goes through LightningCLI which type-checks but doesn't
run cross-field validators. Optional: wire validators into spec path.

### 6. W6: Resume checkpoint probe is side effect (LOW)

`resume.exists()` makes resolver output depend on filesystem state.
Accept as inherently stateful, or extract to a callback.

---

## Phase 3 — Ongoing discipline

- **Scope discipline for `TrainingRunConfig`**: `extra="forbid"` enforced. Review fields quarterly.
- **Recipe generation as code**: evaluate when recipe complexity demands parametric generation.

## Reference

- `docs/decisions/0007-config-system-architecture.md` — canonical design
- `graphids/config/CONFIG_REFERENCE.md` — parameter axes and infrastructure
