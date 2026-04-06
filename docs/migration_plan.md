# Config & Artifact Stack Migration Plan

## Context

**Current pain:** 3-chain YAML + custom resolver + LightningCLI + jsonargparse + file-based artifact catalog rebuilt by DuckDB query.

**Target:** jsonnet (composition) → Typer CLI → Pydantic validation → Lightning Trainer → DuckDB catalog.

**Status (2026-04-06):** Phases 1–5 complete. Phase 6 (PyIceberg) deferred — current DuckDB catalog + run_record.json sidecars are sufficient. Phase 7 not planned.

---

## Dependency Graph

```
Phase 1 (jsonnet)
    └── Phase 2 (Pydantic)
            ├── Phase 3 (strip LightningCLI)
            │       └── Phase 4 (jsonargparse retooling)
            │               └── Phase 5 (Dagster boundaries)
            └── Phase 6 (PyIceberg)  ← independent of 3/4/5, run in parallel
                    └── Phase 7 (sweeps)
```

Phase 6 is fully independent — start it alongside Phase 2 since it only touches artifact writes, not config parsing. Everything else is sequential.

---

## Phase 1 — Jsonnet ✓

**Completed 2026-04-05.** Replaced YAML chain + `merge_yaml_chain` + override
plumbing with jsonnet.

---

## Phase 2 — Pydantic Validation Layer ✓

**Completed 2026-04-05.** `graphids.config.schemas.validate_config` validates
rendered jsonnet output via Pydantic `@model_validator` rules.
---

## Phase 3 — Strip LightningCLI ✓

**Completed 2026-04-05.** LightningCLI removed. `graphids.instantiate.instantiate`
handles class_path import + signature-filtered link_arguments directly.

---

## Phase 4 — Jsonargparse Retooling ✓

**Completed 2026-04-05.** Analyzer configs (`configs/stages/analyze_*.jsonnet`)
use jsonargparse `parser_mode="jsonnet"` via `cli/_analysis.py`.

## Phase 5 — Dagster Asset Config Boundaries ✓

**Completed 2026-04-05.**

`TrainingAssetConfig(dg.Config)` in `orchestrate/dagster/asset_config.py` provides
launch-time overridable knobs (`run_test`, `run_analysis`, `dry_run`).
Asset function returns `dg.Output[str]` with metadata. Checkpoint paths
already flowed via Dagster asset I/O — `upstream_ckpt_paths` in
`TrainingSpec` is populated from asset inputs at resolution time, not from
config. Identity fields stay in `StageConfig` (planner-derived, not
overridable).

---

## Phase 6 — PyIceberg Catalog (deferred)

**Status:** Deferred indefinitely. The current `run_record.json` sidecar +
DuckDB catalog (`rebuild-catalog` command) is sufficient for experiment
tracking. PyIceberg adds complexity without clear benefit at current scale.

---

## Phase 7 — Sweep Integration (not planned)

---

## What Is Kept, Removed, and Added

|                                                        | Status                          |
| ------------------------------------------------------ | ------------------------------- |
| Lightning `Trainer`, DDP, callbacks, `LightningModule` | **Kept**                        |
| DuckDB for querying                                    | **Kept** + run_record sidecars  |
| LightningCLI                                           | **Removed** (Phase 3) ✓        |
| jsonargparse                                           | **Kept** for analyzer (Phase 4) ✓ |
| YAML config chain                                      | **Removed** (Phase 1) ✓        |
| Custom resolver                                        | **Removed** (Phase 1) ✓        |
| `go-jsonnet` binary                                    | **Added** (Phase 1) ✓          |
| Typer CLI                                              | **Added** (replaced argparse)   |
| Pydantic validation layer                              | **Added** (Phase 2) ✓          |
| Dagster asset config boundaries                        | **Added** (Phase 5) ✓          |
| PyIceberg                                              | **Deferred** — not needed       |
