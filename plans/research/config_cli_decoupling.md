# Decoupled CLI Architecture: GraphIDSCLI above LightningCLI

> Discussion from 2026-04-01 session. Phase 3 target architecture.
> Builds on `config_system_synthesis.md` Part 5 (ConfigResolver as exclusive merge path).

---

## Problem Statement

The ConfigResolver (P2.2) validates cross-field constraints but faces an inherent
tension: jsonargparse's full merge + type validation requires importing model classes
(torch, Lightning, PyG), but the resolver runs in contexts where those imports are
undesirable (dagster workers, login node validation).

The current workaround is a naive YAML merge (`_deep_merge` + `_apply_dotted_overrides`)
that reads config files without class imports and checks specific scalar paths. This
works for the 3 known constraints but doesn't scale to arbitrary type-validated merge.

## Proposed Architecture

Decouple `GraphIDSCLI` from `LightningCLI` so the top-level CLI depends only on
jsonargparse (no torch at import time). LightningCLI becomes an internal execution
backend, not the entry point.

### Current

```
GraphIDSCLI(LightningCLI)     <- imports torch at import time
  ├── fit/test/validate/predict  (all Lightning)
  └── __main__.py dispatches operational commands separately

Dagster: builds CLI strings, submits via SLURM
ConfigResolver: naive YAML merge for validation
```

### Proposed

```
GraphIDSCLI (jsonargparse only)  <- no torch at import time
  ├── resolve: merge YAML chain + overrides -> single resolved dict
  │   └── cross-field validation on merged dict
  ├── fit/test: lazy-import LightningCLI, hand it resolved config
  ├── dagster/orchestrate: use resolved dict directly
  └── operational commands: same as today

Flow: resolve() -> write config_snapshot -> LightningCLI --config <snapshot>
```

### Key Principle: Merge Once, Execute Anywhere

The top-level CLI owns config resolution. LightningCLI receives a single
pre-resolved config file — no chaining, no double merge, no divergence.

This is the synthesis doc's flow reframed:
```
ConfigResolver.resolve() -> TrainingRunConfig -> .to_lightning_yaml() -> single file
```

## What It Solves

1. **W1 (YAML-aware validation)** — resolver uses jsonargparse's own merge engine,
   not a naive reimplementation. No import cost because jsonargparse doesn't need
   class imports for dict merging.

2. **W2 (structural exclusivity)** — one merge point, period. LightningCLI sees
   one file, never a chain.

3. **Dagster import constraint** — dagster calls CLI as subprocess via SLURM.
   Resolution happens in CLI process, not dagster process.

4. **"Two merge paths" antipattern** — eliminated. jsonargparse merges once.
   LightningCLI type-validates but doesn't re-merge.

## Concerns

### 1. LightningCLI expects to own the parser

LightningCLI's value is: give me classes, I introspect `__init__` signatures,
build a parser, generate CLI flags. If you pre-resolve configs, you bypass most
of that machinery. It becomes a glorified:
```python
Trainer(**config["trainer"]) + Model(**config["model"]["init_args"])
```

Counter: this is fine. The CLI generation is useful for dev/test (ad hoc overrides).
For pipeline runs, the parser machinery is wasted work.

### 2. `class_path`/`init_args` serialization boundary

The pre-resolved config must emit the structure LightningCLI expects. Any Lightning
upgrade that changes the format breaks `to_lightning_yaml()`. Mitigation: keep the
boundary narrow (10-20 fields, not hundreds).

### 3. Dev/test ergonomics

Today:
```bash
python -m graphids fit --config stages/autoencoder.yaml \
  --config models/vgae/scales/small.yaml --model.init_args.lr=0.01
```

This composes directly via jsonargparse. With the decoupled CLI, two options:
- **Passthrough mode:** detect dev invocation (multiple `--config` flags, no recipe),
  delegate directly to LightningCLI. Dev path unchanged.
- **Resolve-first mode:** always resolve, then execute. Uniform but adds a step.

Passthrough mode is simpler and preserves existing UX.

### 4. Synthesis doc warning

Part 6: "Don't add a parallel merge path." If the top-level CLI merges with
jsonargparse and LightningCLI also merges, that's two paths. Mitigation: the
top-level merge produces a single file; LightningCLI reads one file with no
chaining. LightningCLI's "merge" becomes a no-op (single input, nothing to merge).

## Comparison with Alternatives

| Approach | Validates | Import cost | Complexity | Status |
|---|---|---|---|---|
| **Naive YAML merge** | 3 specific constraints | None | ~30 lines | Implemented (P2.2) |
| **Decoupled CLI** | Full merged state | None (jsonargparse only) | ~200 lines + refactor | Phase 3 target |
| **jsonargparse in resolver** | Full + type-validated | torch+Lightning+PyG | Low code, high import | Rejected (import cost) |
| **Dynaconf** | None (first-level merge only) | Low | New dependency | Rejected (see config_tool_comparison.md) |
| **OmegaConf** | Structured merge | Low | New dep + dual system | Not recommended (see config_tool_comparison.md) |

## Implementation Sketch

```python
# graphids/cli.py — top-level, no torch import
from jsonargparse import ArgumentParser

class GraphIDSCLI:
    """Top-level CLI. Owns config resolution. Delegates execution."""

    def resolve_config(
        self,
        config_files: list[str],
        overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Merge YAML chain using jsonargparse, apply overrides. No type validation."""
        parser = ArgumentParser()
        # jsonargparse can merge configs without class args defined
        # ...
        return merged_dict

    def fit(self, config_files, overrides=None):
        """Resolve, snapshot, then lazy-import LightningCLI for execution."""
        resolved = self.resolve_config(config_files, overrides)
        snapshot_path = self._write_snapshot(resolved)

        # Lazy import — torch loaded here, not at CLI import
        from pytorch_lightning.cli import LightningCLI
        # LightningCLI reads single pre-resolved config
        cli = LightningCLI(args=["fit", "--config", str(snapshot_path)])
```

## Decision

**Phase 3 target.** The naive YAML merge solves the immediate problem (3 constraints).
The decoupled CLI is the architecturally correct long-term solution but requires:
- Rethinking the entry point and __main__.py dispatch
- Verifying jsonargparse can merge configs without class-aware parsers
- Designing the dev/test passthrough mode
- Testing that single-file LightningCLI execution matches multi-config behavior

Revisit after the ablation run surfaces real pain points with the current approach.

## References

- `config_system_synthesis.md` Part 5 — ConfigResolver design, `to_lightning_yaml()`
- `config_tool_comparison.md` — Dynaconf (rejected), OmegaConf (not recommended)
- `issues/config-system-overhaul.md` — W1-W7 weakness list
- `feedback_dagster_no_torch_import.md` — import constraint origin
