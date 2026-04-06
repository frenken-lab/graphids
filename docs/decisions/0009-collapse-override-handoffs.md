# ADR 0009: Collapse Override Handoff Chain (9 → 3)

## Context

The original pipeline passed stringified override dicts through multiple
transport steps and only validated inside the SLURM job. That created a
validation desert between planning and execution, with typos or malformed KD
payloads surfacing only after queue wait + GPU startup.

## Decision

Collapse the handoff chain by transporting only `TrainingSpec`
(`jsonnet_path` + typed `jsonnet_tla`) and validating at both planning time and
on the worker:

1. `ConfigResolver.resolve()` builds `jsonnet_tla` via
   `graphids.orchestrate.contracts.build_tla_dict`, then calls
   `render_config(...)` and `validate_config(...)`.
2. The SLURM job (`from-spec`) re-renders and re-validates before
   instantiating the Lightning stack directly.

## Target chain

```
1. Recipe YAML → expand_recipe_configs() → enumerate_assets()
2. StageConfig → ConfigResolver.resolve()
   └ build_tla_dict → render_config → validate_config → cross-field rules
3. TrainingSpec → to_envelope → sbatch
4. SLURM job → from-spec → render_config → validate_config → instantiate
```

## Consequences

- Override key typos fail before submission.
- KD auxiliary payloads remain typed; no JSON/YAML string round-trip.
- Pipeline path no longer constructs CLI args or relies on jsonargparse.

## Sources

- `graphids/orchestrate/resolve/resolver.py`
- `graphids/orchestrate/contracts/__init__.py`
- `graphids/core/train_entrypoint.py`
- `docs/reference/config-architecture.md`
