# ADR 0009: Collapse Override Handoff Chain (9 → 3)

## Context

The override pipeline from recipe YAML to training execution has 9 handoffs.
Stages 4-8 are pure transport with zero validation — a "validation desert"
between asset enumeration and jsonargparse. Override key typos pass through
as untyped strings and only fail inside the SLURM job, after queue wait +
GPU startup.

### Current chain

```
1. Recipe YAML → expand_recipe_configs()      Pydantic: shape + enums
2. expand output → enumerate_assets()          Pydantic: identity fields
3. StageConfig → ConfigResolver.resolve()      Cross-field constraints
4. ResolvedConfig → Dagster asset              Serialization (no validation)
5. JSON spec → sbatch                          Shell (no validation)
6. SLURM job → from-spec --phase train          Deserialization (no validation)
7. TrainingSpec → to_override_dict()           Dict construction (no validation)
8. override dict → merge_yaml_chain()          YAML merge (no validation)
9. CLI args → jsonargparse                     FULL validation
```

### Leakage

- **Override key typos** — `trainer.max_epoch` passes stages 1-8, fails at 9.
- **KD JSON blob** — `json.dumps([kd_overrides])` stuffed into runtime_overrides
  as a string. Never validated until stage 9.
- **to_override_dict → _build_cli_args round-trip** — dict → CLI strings →
  jsonargparse parses back to dict. Any prefixing bug creates bad CLI args
  that fail at stage 9.

## Decision

Collapse to 3 handoffs by:

**A. Move `parser.parse_object()` into ConfigResolver.resolve().**

Currently `parse_object()` exists only in the `validate` command (optional,
manual). Move it into the resolver so every submitted job is pre-validated
automatically. Override key typos die at planning time.

```python
# In ConfigResolver.resolve(), after building runtime_overrides:
merged = merge_yaml_chain(cfg.config_files, runtime_overrides)
parser.parse_object(merged)  # raises on bad keys/types
```

Requires a parser instance in ConfigResolver. The parser can be created
once with `GraphIDSCLI(run=False)` and reused. This import is lazy
(torch loaded only when parser is first needed), and only happens on the
dagster worker (CPU SLURM job), not at dagster definition time.

**B. Replace CLI string round-trip with direct dict instantiation on SLURM side.**

Instead of:
```python
# Current (train_entrypoint.py):
overrides = TrainingContract.to_override_dict(spec)      # dict
resolved = merge_yaml_chain(config_files, overrides)     # dict
args = _build_cli_args(spec)                             # list[str]
run_lightning(args)                                      # jsonargparse parses back to dict
```

Do:
```python
# Proposed:
overrides = TrainingContract.to_override_dict(spec)      # still needed for ckpt flags
resolved = merge_yaml_chain(config_files, overrides)
parser = GraphIDSCLI(run=False).parser
cfg = parser.parse_object(resolved)                      # validate + type coerce
trainer, model, data = parser.instantiate_classes(cfg)    # no CLI round-trip
trainer.fit(model, datamodule=data)
```

Dev path (`python -m graphids fit --config ...`) keeps CLI args unchanged.
Pipeline path uses dict-based instantiation.

### Target chain

```
1. Recipe YAML → Plan                         Pydantic + parse_object() validates ALL
    ├ expand_recipe_configs()                  keys, types, enums, cross-fields
    ├ enumerate_assets()                       in one pass at planning time
    └ ConfigResolver.resolve()
                                               ── serialization boundary (JSON) ──
2. SLURM job → Run training                   Deserialize spec, merge configs,
    ├ merge_yaml_chain()                       instantiate directly via parser
    └ parser.instantiate_classes()             (no CLI string round-trip)
```

## What this deletes

| Code | Lines | Why deletable |
|------|-------|---------------|
| `_build_cli_args()` | 9 | Pipeline path uses dict instantiation |
| LINK_TARGETS replay in train_entrypoint.py | 15 | `parse_object()` handles links |
| `validate` command's parse loop | ~30 | Validation now happens in resolver |
| Redundant jsonargparse parse at stage 9 | 0 | Still runs (safety net), but no longer the *only* gate |

## What this requires verifying

1. **`parser.instantiate_classes()` + forced callbacks.** `GraphIDSCLI.add_arguments_to_parser()`
   registers ModelCheckpoint, EarlyStopping, LearningRateMonitor via `parser.add_lightning_class_args()`.
   Verify these are present in `parse_object()` output, not only when parsing CLI args.

2. **`before_instantiate_classes()` hooks.** GraphIDSCLI patches logger save_dirs and
   checkpoint dirpath in this hook. Verify `instantiate_classes()` triggers it, or
   replicate the patches in the pipeline path.

3. **LINK_TARGETS in dict path.** `parser.link_arguments()` may only fire during
   `parse_args()`, not `parse_object()`. Verify or apply links manually (as today).

4. **All model types.** Run smoke test across autoencoder, normal, curriculum, fusion
   to verify dict-based instantiation produces identical training behavior.

## Consequences

- Every SLURM job is pre-validated at planning time. Key typos never reach the queue.
- KD JSON blob is validated by `parse_object()` at planning time.
- Pipeline path is ~25 lines shorter (no CLI string construction).
- Dev path is unchanged — still uses CLI args.
- Two entry points diverge slightly: CLI path uses `parse_args`, pipeline path uses
  `parse_object` + `instantiate_classes`. Divergence is acceptable because both go
  through the same jsonargparse parser with the same registered types.

## Sources

- jsonargparse v4.47.0 docs: `parse_object()`, `instantiate_classes()`, `validate()`
- `graphids/orchestrate/validate.py` — existing `parse_object()` usage (this session)
- `graphids/core/train_entrypoint.py` — current CLI round-trip
- `graphids/_lightning.py:206-252` — GraphIDSCLI forced callbacks + hooks
