# ADR 0007: Config System Architecture — Patterns and Diagnosis

> Date: 2026-03-31 | Status: **Accepted** | Canonical config design reference

## Context

Config combinatorial explosion was causing real bugs: lost checkpoints (list replacement),
quadratic file growth (cross-product encoding), silent drift (parallel topology declarations).
Surveyed 15+ production systems to identify patterns and match them to KD-GAT's needs.

## Three Patterns for Config Composition

### Pattern 1: Hierarchical Composition (Defaults Lists)
Primary config names which option from each axis. Each axis = directory, each option = file.
File count scales linearly. **Production:** Hydra, Habitat Lab, Fairseq, NeMo.
**Failure mode:** breaks when axes aren't independent.

### Pattern 2: Base + Overlay (Delta Patches)
Complete base + thin overlays. Max depth ~3 levels. **Production:** MMDetection, Kustomize, Helm.
**Failure mode:** list replacement is #1 trap — naive deep-merge replaces lists atomically.

### Pattern 3: Programmatic Config (Code-as-Config)
Config in a real language, separate `instantiate()` step. Sub-linear file growth.
**Production:** Detectron2 LazyConfig, Fiddle, Jsonnet. **Failure mode:** higher learning curve.

## Diagnosis: KD-GAT Was an Accidental Hybrid

| Problem | Pattern | Severity | Resolution |
|---|---|---|---|
| List replacement drops callbacks | P2 failure | Critical | Forced callbacks (`add_lightning_class_args`) |
| Cross-product encoding (scale×model in one file) | P1 manual | Structural | Independent axes (`models/{family}/scales/`) |
| Parallel topology declarations | Neither | Moderate | `topology.py` import-time cross-validation |
| Manual recipe enumeration | Manual | Moderate | `expand_recipe_configs()` |
| Three config domains with no contract | Structural | Moderate | `TrainingRunConfig` + `ConfigResolver` |

## Decision: Two-Level Fix

**Level 1 (YAML restructuring, DONE):** Independent config axes, forced callbacks,
import-time validation, directory reorganization. Zero new code.

**Level 2 (Narrow typed contract, DONE):** `TrainingRunConfig` (Pydantic, `extra="forbid"`)
for boundary parameters. `ConfigResolver` for cross-field validation + audit.
`PathContext` (frozen) for write path enforcement.

## What NOT to Do

1. Don't adopt Hydra — jsonargparse multi-`--config` is equivalent
2. Don't mirror every `__init__` signature — TrainingRunConfig is narrow (10-20 params)
3. Don't add a parallel merge path — ConfigResolver replaces, not layers
4. Don't template YAML — independent axes solve it within plain YAML
5. Don't maintain parallel topology declarations — code reads YAML, no parallel enums

## Sources

- [Hydra defaults list](https://hydra.cc/docs/advanced/defaults_list/)
- [MMDetection config docs](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html)
- [Kustomize docs](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/)
- [Config Complexity Curse](https://blog.cedriccharly.com/post/20191109-the-configuration-complexity-curse/)
