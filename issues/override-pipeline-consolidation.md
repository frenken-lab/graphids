# Override pipeline consolidation

## Problem

Overrides flow through 4 hops between recipe YAML and the final CLI args
passed to LightningCLI. Each hop transforms keys/values slightly, and a bug
at any hop silently propagates to downstream hops.

### The 4-hop flow

```
Recipe YAML
    |
    v
(1) _flatten_dict()          -- recipe_expand.py:56
    Flattens nested dict to dotted keys: {"trainer": {"max_epochs": 2}}
    becomes {"trainer.max_epochs": "2"}. Stringifies all values.
    Output: stored in expanded recipe as trainer_overrides / model_overrides.
    |
    v
(2) runtime_overrides         -- resolve.py:85
    ConfigResolver.resolve() copies trainer_overrides and kd_overrides into
    a flat runtime_overrides dict. Also injects curriculum epoch sync and
    resume checkpoint. Passes the dict into TrainingSpec.runtime_overrides.
    |
    v
(3) to_override_dict()       -- contracts/ops.py:124
    TrainingContract.to_override_dict() re-derives the full CLI override dict
    from TrainingSpec fields: adds "model.init_args." prefix to model_init_overrides,
    resolves upstream checkpoint flags, then merges runtime_overrides on top.
    |
    v
(4) _build_cli_args()        -- train_entrypoint.py:19
    Converts the override dict to ["--key=value", ...] list for LightningCLI.
```

### The double-prefix bug (concrete example)

If `_flatten_dict` in step 1 produces `"model.init_args.lr": "0.01"` and
this ends up in `model_init_overrides` (which already has its own prefix logic),
step 3 re-prefixes it to `"model.init_args.model.init_args.lr"`. The value
silently becomes an unknown key rejected by jsonargparse -- or worse, ignored.

This happened because the boundary between "what has a prefix" and "what
doesn't" is implicit. Each hop assumes different prefix conventions:
- `_flatten_dict` adds the prefix based on the nesting.
- `runtime_overrides` passes keys through as-is (already prefixed).
- `to_override_dict` adds `model.init_args.` to model_init_overrides.
- `_build_cli_args` adds `--` to everything.

There is no single place that defines "this key is fully qualified" vs
"this key needs a prefix."

### Additional fragility

- **Type coercion diverges.** `_flatten_dict` lowercases bools. `_cli_scalar`
  (in ops.py) also lowercases bools. If both are applied, it's fine, but the
  duplication means they can drift.
- **runtime_overrides is a grab bag.** It contains trainer overrides (prefixed),
  KD overrides (prefixed), resume checkpoint paths, and curriculum sync
  entries. No schema, no typing.
- **Testing is per-hop.** `test_build_cli_args_*` tests (test_pure.py) only
  test hop 4. There are no integration tests that trace a recipe override
  all the way from YAML to CLI args.

## Proposed consolidation

### Single `OverrideChain` class

```python
@dataclass
class OverrideChain:
    """Traces an override from recipe origin to CLI arg.

    All keys are stored fully-qualified (e.g. "model.init_args.lr").
    No implicit prefix addition -- callers must provide full keys.
    """
    # Immutable record of all overrides with provenance
    entries: tuple[OverrideEntry, ...]

    @classmethod
    def from_recipe(cls, recipe_overrides: dict, trainer_overrides: dict) -> OverrideChain:
        """Single entry point: recipe dict -> fully-qualified overrides."""
        ...

    def to_cli_args(self) -> list[str]:
        """Terminal conversion: override chain -> ["--key=value", ...]."""
        ...

    def to_override_dict(self) -> dict[str, str]:
        """For cross-field validation (merge_yaml_chain input)."""
        ...

@dataclass(frozen=True)
class OverrideEntry:
    key: str           # fully qualified: "trainer.max_epochs"
    value: str         # stringified
    source: str        # "recipe_trainer", "kd", "resume_ckpt", etc.
```

### Key design decisions

1. **All keys are fully-qualified at entry.** `_flatten_dict` already produces
   fully-qualified keys. Model init overrides must be stored as
   `"model.init_args.X"` not just `"X"`. This eliminates the prefix ambiguity.

2. **`OverrideChain` replaces `runtime_overrides` dict.** Instead of a bare dict
   inside TrainingSpec, the chain carries provenance. The existing
   `OverrideRecord` in resolve.py (used for audit logging) merges into
   `OverrideEntry`.

3. **`to_override_dict` and `to_cli_args` are the only output methods.**
   Replaces the separate `TrainingContract.to_override_dict` and
   `_build_cli_args` functions. The chain is the single source of truth.

4. **Cross-field validation reads from the chain.** `ConfigResolver` validates
   against `chain.to_override_dict()` merged with the YAML configs, same as
   today but with a cleaner interface.

### Migration path

1. Add `OverrideChain` + `OverrideEntry` to `graphids/core/contracts/`.
2. Have `ConfigResolver.resolve()` build an `OverrideChain` instead of a
   bare `runtime_overrides` dict. The existing `OverrideRecord` audit list
   becomes the chain's entries.
3. `TrainingSpec.runtime_overrides` becomes a serialized `OverrideChain`
   (or the chain serializes to the existing dict format for backward compat
   with in-flight SLURM jobs).
4. `_build_cli_args` delegates to `chain.to_cli_args()`.
5. `TrainingContract.to_override_dict` delegates to `chain.to_override_dict()`.
6. Delete `_flatten_dict` prefix logic (chain handles it at construction).
7. Add integration test: recipe YAML -> OverrideChain -> CLI args, asserting
   no double-prefixes and round-trip fidelity.

### Files affected

- `graphids/core/contracts/` -- new OverrideChain/OverrideEntry
- `graphids/config/recipe_expand.py` -- _flatten_dict feeds into chain
- `graphids/orchestrate/resolve.py` -- ConfigResolver builds chain
- `graphids/core/contracts/ops.py` -- to_override_dict delegates to chain
- `graphids/core/train_entrypoint.py` -- _build_cli_args delegates to chain
- `tests/orchestrate/test_pure.py` -- integration test recipe->CLI

---

## Mitigations applied (2026-04-01, session 7)

The 4-hop architecture is unchanged, but several guards now catch the failure
modes this issue describes:

1. **`_flatten_dict` rejects non-scalars** — `TypeError` on lists/dicts prevents
   structured values from being silently `str()`-ified (`recipe_expand.py`).
2. **`merge_yaml_chain` raises on missing files** — `FileNotFoundError` instead of
   silent skip prevents typos in config paths from producing default-only configs
   (`yaml_utils.py`).
3. **`to_override_dict` warns on key conflicts** — `runtime_overrides_clobber`
   structlog warning when `runtime_overrides` collides with `model_init_overrides`
   (`contracts/ops.py`).
4. **Unmapped upstream models raise** — `KeyError` instead of silent drop when
   `_CKPT_FLAG_BY_MODEL` has no entry for a model family (`contracts/ops.py`).
5. **Config snapshot includes LINK_TARGETS** — snapshot YAML is now reproducible
   for manual replay (`train_entrypoint.py`).

The `OverrideChain` proposal remains the architectural fix. These guards reduce
the blast radius of the existing 4-hop flow until it's consolidated.
