# Override pipeline consolidation

## Problem

Overrides flow through 4 hops between recipe YAML and final CLI args. Each hop
transforms keys/values slightly, and a bug at any hop silently propagates.

```
Recipe YAML → (1) _flatten_dict() → (2) runtime_overrides → (3) to_override_dict() → (4) _build_cli_args()
```

Key fragility: the boundary between "fully-qualified key" and "needs prefix" is
implicit. `_flatten_dict` produces prefixed keys, `to_override_dict` re-prefixes
`model_init_overrides` — double-prefix bugs are possible.

## Mitigations Applied (2026-04-01)

5 guards now catch the main failure modes:

1. `_flatten_dict` rejects non-scalars (`TypeError`)
2. `merge_yaml_chain` raises on missing config files (`FileNotFoundError`)
3. `to_override_dict` warns on key conflicts (`runtime_overrides_clobber`)
4. Unmapped upstream models raise `KeyError` (not silent drop)
5. Config snapshot includes `LINK_TARGETS` for manual replay

## Proposed Architectural Fix: `OverrideChain`

A single `OverrideChain` class replaces the 4-hop flow:

- All keys stored fully-qualified at entry (no implicit prefix addition)
- Carries provenance per entry (source: recipe_trainer, kd, resume_ckpt, etc.)
- `to_cli_args()` and `to_override_dict()` are the only output methods
- Replaces `runtime_overrides` dict in `TrainingSpec`
- Merges existing `OverrideRecord` (resolve.py audit) into `OverrideEntry`

### Files affected

`core/contracts/` (new types), `config/recipe_expand.py`, `orchestrate/resolve.py`,
`core/contracts/ops.py`, `core/train_entrypoint.py`, `tests/orchestrate/test_pure.py`

**Status:** Mitigations reduce blast radius. `OverrideChain` is the architectural fix
when override complexity warrants it.
