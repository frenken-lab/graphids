# ADR 0001: Reject Hydra/OmegaConf, Keep jsonargparse + Naive YAML Merge

> **Stale reference note (2026-04-06):** LightningCLI was removed in Phase 3. Config is now jsonnet → Pydantic validate → `graphids.instantiate.instantiate()`. `cli.py` and `_lightning.py` no longer exist; CLI is Typer-based in `graphids/cli/`.

> Date: 2026-04-01 | Status: **Accepted**

## Context

KD-GAT's config system uses LightningCLI (jsonargparse) for training and dagster for
orchestration. The config resolution layer merges 3-4 YAML files with dotted-key overrides.
We evaluated whether Hydra, OmegaConf, Dynaconf, or a decoupled CLI architecture could
improve this.

## Decision

**Keep jsonargparse + naive YAML merge (~30 lines). Reject all alternatives.**

## Rationale

### Hydra: Rejected
jsonargparse's multi-`--config` composition is equivalent to Hydra's defaults lists.
Switching loses tight Lightning integration (`link_arguments`, `add_lightning_class_args`,
`subclass_mode`). OmegaConf's lazy resolution reintroduces unpacking issues.

### OmegaConf (standalone): Not Adopted
Does the merge correctly, but our ~15-line `deep_merge` already works. Adding OmegaConf
creates DictConfig impedance (not a plain dict) and a dual-merge anti-pattern (OmegaConf
for dagster path, jsonargparse for dev path).

### Dynaconf: Rejected
First-level-only YAML merge is a fundamental blocker. Our configs are 3-4 levels deep.

### Decoupled CLI (GraphIDSCLI above LightningCLI): Tried and Reverted
Built 80 lines of direct Model/Data/Trainer instantiation. Immediate drift risk — every
wiring detail duplicated between `_lightning.py` and `train_entrypoint.py`. Replaced with
`train_entrypoint.py` building CLI args from `TrainingSpec` and calling `run_lightning()`.
LightningCLI handles merge, type validation, instantiation. Don't fight the framework.

## Consequences

- ConfigResolver role is cross-field validation + audit, not merge-for-instantiation
- Both dev and pipeline paths converge at `run_lightning()` → `GraphIDSCLI(LightningCLI)`
- Naive merge divergence from jsonargparse is a known risk, mitigated by `test_merge_parity.py`
- `cli.py` is torch-free; `_lightning.py` is lazy-imported at execution time

## When to Reconsider

- If interpolation (`${key}`) becomes needed → OmegaConf
- If Hydra is adopted for sweeps → reconsider full stack
- If list merge control needed beyond forced callbacks → OmegaConf `ListMergeMode`

## Sources

- `docs/decisions/0007-config-system-architecture.md` — pattern analysis
- [jsonargparse docs](https://jsonargparse.readthedocs.io/)
- [Hydra defaults list](https://hydra.cc/docs/advanced/defaults_list/)
- [OmegaConf usage](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html)
- [Lightning #15109](https://github.com/Lightning-AI/pytorch-lightning/issues/15109)
