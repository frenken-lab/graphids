# Config Tool Comparison: Dynaconf vs OmegaConf

> Research for KD-GAT config resolution layer.
> Question: could either tool serve as "merge once, validate, then hand resolved config to LightningCLI"?
>
> Date: 2026-04-01
> Context: `config_system_synthesis.md` Part 5 defines a `ConfigResolver` that merges YAML chain,
> applies overrides, validates, and emits a single resolved config. The interim implementation uses
> bare `yaml.safe_load` + `dict.update` (~30 lines). This document evaluates whether a config library
> adds value over that approach.

---

## 1. Dynaconf (v3.2.13, March 2026)

**What it is:** Python settings management library. Designed for 12-factor apps (Django/Flask),
environment layering, secrets vaults. 4.3k GitHub stars, actively maintained.

**Dependencies:** Lightweight. Core has no heavy deps. YAML support via optional `pyyaml`.
Redis/Vault integrations are optional extras.

### 1.1 YAML Chaining

Dynaconf loads multiple files via `settings_files=["a.yaml", "b.yaml"]` or glob patterns.
Files are loaded in order. **Default behavior is replace, not merge** -- later files overwrite
earlier keys at the top level.

Deep merge requires explicit opt-in via one of:
- `dynaconf_merge: true` in a YAML file (marks entire file for merge)
- `dynaconf_merge` token appended to list values
- Dunder notation (`DATABASE__PASSWORD=1234`) for env var overrides
- Global `merge_enabled=True` (warned against in docs: "can lead to unexpected results")

**Critical limitation:** `dynaconf_merge` and `@merge` "work only for the first level keys, it
will not merge subdicts or nested lists (yet)." For deep nesting, dunder notation is recommended --
but that only works for env vars and dotlist overrides, not YAML-to-YAML merge.

Source: [Merging docs](https://www.dynaconf.com/merging/),
[Settings files](https://www.dynaconf.com/settings_files/)

**Verdict for KD-GAT:** Inadequate. Our configs are 3-4 levels deep (`model.init_args.hidden_dims`).
First-level-only merge means stage YAML would clobber model base YAML's `model:` dict entirely,
losing `class_path`. This is the exact list-replacement problem diagnosed in
`config_system_synthesis.md` Part 3, now applied to dicts too.

### 1.2 Override Resolution

Dynaconf has a fixed priority chain:
1. Default values
2. Settings files (layered by environment)
3. Environment variables (prefixed, e.g. `DYNACONF_TRAINER__MAX_EPOCHS=2`)

Env vars use dunder (`__`) for nesting: `DYNACONF_MODEL__INIT_ARGS__LR=0.001`.
No native CLI argument parsing -- it reads env vars, not `sys.argv`.

Source: [Dynaconf home](https://www.dynaconf.com/)

**Verdict:** Env-var override works. CLI override absent -- we'd still need argparse or jsonargparse
for `--trainer.max_epochs=2`. Two override systems = complexity for no gain.

### 1.3 Type Validation

Dynaconf provides a `Validator` class with explicit rules:

```python
Validator("AGE", gte=20, lte=80)
Validator("NAME", must_exist=True)
Validator("DB.PORT", is_type_of=int)
```

Cross-field validation supported via `when` parameter:
```python
Validator("DATABASE.HOST", must_exist=True,
    when=Validator("DATABASE.USER", must_exist=True))
```

**No support for Python type hints, dataclasses, or Pydantic models** as schema source.
Every validation rule must be written manually. No automatic schema extraction from `__init__`
signatures.

Source: [Validation docs](https://www.dynaconf.com/validation/)

**Verdict:** Inferior to what we already have. `TrainingRunConfig` (Pydantic, `extra="forbid"`) +
jsonargparse type checking from `__init__` signatures already provides stronger validation with less
code.

### 1.4 class_path / init_args

Dynaconf has no concept of deferred instantiation, class registries, or `class_path`/`init_args`
patterns. It treats all YAML values as plain data. A config like:

```yaml
model:
  class_path: graphids.core.models.autoencoder.vgae.VGAEModule
  init_args:
    lr: 0.002
```

would be loaded as a plain dict with string values. No instantiation, no type checking of
`init_args` against the class's `__init__` signature. The `class_path` string would just be
a string -- Dynaconf wouldn't know it refers to a class.

**Verdict:** Configs would need no restructuring (Dynaconf preserves arbitrary YAML structure),
but the tool adds nothing to the `class_path`/`init_args` workflow. Pure passthrough.

### 1.5 Import Weight

Dynaconf itself is lightweight -- no torch, no Lightning imports. Config resolution can happen
without importing model classes.

**Verdict:** Good. Meets the "no torch at resolution time" requirement.

### 1.6 Dagster Compatibility

No known integration. No conflicts either -- Dynaconf is just a settings loader. Dagster has
its own Pydantic-based `Config` system. Using both means maintaining two config systems.

Source: [Dagster config docs](https://docs.dagster.io/concepts/configuration/config-schema)

### 1.7 LightningCLI Compatibility

No integration. LightningCLI uses jsonargparse internally. Dynaconf would be a separate layer
that loads/merges YAML, then... what? Emits a dict that gets written to a temp YAML file for
LightningCLI to re-parse? Or bypasses LightningCLI entirely? Neither is clean.

---

## 2. OmegaConf (v2.3.0 stable, Dec 2022; v2.4.0.dev3, Sep 2025)

**What it is:** Hierarchical config system. The backend for Hydra. Deep merge, structured
config validation, interpolation. 1.9k GitHub stars.

**Dependencies:** `antlr4-python3-runtime`, `pyyaml`. No torch, no Lightning. Lightweight.

### 2.1 YAML Chaining

`OmegaConf.merge()` performs **true deep merge** with last-value-wins:

```python
conf1 = OmegaConf.load("base.yaml")       # {model: {class_path: "...", init_args: {lr: 0.01}}}
conf2 = OmegaConf.load("scale.yaml")      # {model: {init_args: {hidden_dims: [64, 32]}}}
merged = OmegaConf.merge(conf1, conf2)
# Result: {model: {class_path: "...", init_args: {lr: 0.01, hidden_dims: [64, 32]}}}
```

Dicts merge recursively. Lists replace by default (same as jsonargparse). List merge modes
available since 2.2: `REPLACE`, `EXTEND`, `EXTEND_UNIQUE`.

Source: [OmegaConf usage](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html),
[GitHub README](https://github.com/omry/omegaconf)

**Verdict:** Correct deep-merge semantics. Equivalent to what a 15-line `deep_merge()` function
does, but battle-tested across the Hydra ecosystem.

### 2.2 Override Resolution

`OmegaConf.from_dotlist()` parses CLI-style overrides:

```python
overrides = OmegaConf.from_dotlist(["trainer.max_epochs=2", "model.init_args.lr=0.001"])
final = OmegaConf.merge(base, overrides)
```

`OmegaConf.from_cli()` reads directly from `sys.argv`.

Priority is caller-controlled -- `merge()` takes varargs and applies left-to-right:
```python
resolved = OmegaConf.merge(base, stage, scale, experiment_overrides, cli_overrides)
```

This matches the `ConfigResolver` priority chain in `config_system_synthesis.md` Part 5 exactly.

Source: [OmegaConf usage](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html)

**Verdict:** Native support for the exact override pattern we need. `from_dotlist` handles
the `trainer_overrides` flattened dict directly.

### 2.3 Type Validation

Two modes:

**a) Structured configs (dataclass schema):**
```python
@dataclass
class TrainerConfig:
    max_epochs: int = 300
    precision: str = "16-mixed"

@dataclass
class Schema:
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: Any = MISSING  # passthrough -- no type enforcement

schema = OmegaConf.structured(Schema)
loaded = OmegaConf.load("config.yaml")
merged = OmegaConf.merge(schema, loaded)  # ValidationError if trainer.max_epochs="foo"
```

Type enforcement happens at merge time. Assigning `"foo"` to an `int` field raises
`ValidationError`. The schema dataclass **does not need to import torch or model classes** --
it can use `Any` for passthrough sections and typed fields only for sections it wants to validate.

Source: [Structured configs](https://omegaconf.readthedocs.io/en/latest/structured_config.html)

**b) Struct flag (reject unknown keys):**
Structured configs automatically reject access to undefined fields:
```python
conf = OmegaConf.structured(TrainerConfig)
conf.nonexistent  # raises AttributeError
```

This is non-recursive by default -- only top-level keys are protected.

**Verdict:** The selective validation pattern (type-check `trainer` fields, passthrough `model`
and `data` as `Any`) is exactly what we need. Schema dataclasses live in config/ with no torch
imports. Stronger than bare `yaml.safe_load` + Pydantic post-validation, because type errors
are caught *during merge* not after.

### 2.4 class_path / init_args

OmegaConf has **no native concept of `class_path`/`init_args`**. It treats them as plain dict
keys. Hydra's `_target_`/instantiate pattern is similar but lives in Hydra, not OmegaConf.

However, OmegaConf doesn't *interfere* with the pattern either. A config containing:
```yaml
model:
  class_path: graphids.core.models.autoencoder.vgae.VGAEModule
  init_args:
    lr: 0.002
```
loads as a `DictConfig` with those exact keys. `OmegaConf.to_container(cfg, resolve=True)`
converts it to a plain dict that jsonargparse/LightningCLI can consume directly.

**Verdict:** Transparent passthrough. No restructuring needed. OmegaConf merges the YAML
faithfully; LightningCLI interprets `class_path`/`init_args` downstream.

### 2.5 Import Weight

Zero heavy deps. `antlr4-python3-runtime` + `pyyaml` only. Config resolution happens with
no torch, no Lightning, no model class imports.

`OmegaConf.to_container(cfg, resolve=True)` emits a plain `dict` -- no OmegaConf types leak
downstream.

**Verdict:** Meets the requirement cleanly.

### 2.6 Dagster Compatibility

No official integration, no conflicts. OmegaConf resolves config to a plain dict; dagster's
`Config` classes (Pydantic-based) consume plain dicts. The handoff is:
```python
resolved_dict = OmegaConf.to_container(merged, resolve=True)
# Pass resolved_dict to dagster op config or TrainingSpec
```

Dagster never sees OmegaConf types.

### 2.7 LightningCLI Compatibility

jsonargparse already has an OmegaConf integration mode (`parser_mode="omegaconf"`) for
interpolation support. However, this is about using OmegaConf *inside* jsonargparse, not
*replacing* it.

For the "merge before LightningCLI" pattern:
```python
# ConfigResolver using OmegaConf
configs = [OmegaConf.load(f) for f in yaml_chain]
overrides = OmegaConf.from_dotlist(cli_overrides)
merged = OmegaConf.merge(*configs, overrides)
resolved = OmegaConf.to_container(merged, resolve=True)

# Write resolved config, hand to LightningCLI
Path("/tmp/resolved.yaml").write_text(yaml.dump(resolved))
cli = GraphIDSCLI(args=["fit", "--config", "/tmp/resolved.yaml"])
```

Or skip the temp file and pass the dict directly via jsonargparse's API.

Source: [Lightning-AI/pytorch-lightning#15109](https://github.com/Lightning-AI/pytorch-lightning/issues/15109)

**Verdict:** Clean coexistence. OmegaConf handles merge+validate; LightningCLI handles
`class_path` instantiation and CLI generation for dev/test.

---

## 3. Comparison Table

| Dimension | Dynaconf | OmegaConf | Naive YAML merge (status quo) |
|---|---|---|---|
| **Deep merge** | First-level only for YAML; dunder for env vars | True recursive deep merge | Custom `deep_merge()` (~15 lines) |
| **CLI overrides** | Env vars only (dunder notation) | `from_dotlist(["k=v"])` native | Manual `dict.update` on flattened keys |
| **Type validation** | Manual `Validator()` rules | Structured config dataclasses | Pydantic `TrainingRunConfig` post-merge |
| **Schema source** | Explicit Validator objects | Python dataclasses (no model imports) | Pydantic model (no model imports) |
| **class_path/init_args** | Passthrough (no awareness) | Passthrough (no awareness) | Passthrough |
| **Import weight** | Lightweight (no torch) | Lightweight (antlr4 + pyyaml) | Zero deps |
| **Dagster compat** | No integration, no conflict | No integration, no conflict | N/A |
| **LightningCLI compat** | No integration | `parser_mode="omegaconf"` exists | N/A |
| **Merge-time validation** | No (validates on access) | Yes (structured merge raises immediately) | No (validates after merge via Pydantic) |
| **List merge control** | `dynaconf_merge` token (limited) | `ListMergeMode` enum (REPLACE/EXTEND/EXTEND_UNIQUE) | Caller-defined |
| **Interpolation** | `@format` tokens, env var expansion | `${key}` resolver syntax, custom resolvers | None |
| **Maturity for ML** | Web/DevOps focus; limited ML adoption | Backbone of Hydra; dominant in ML research | N/A |
| **Dependencies added** | `dynaconf` | `omegaconf` (antlr4-python3-runtime, pyyaml) | None |

---

## 4. The Key Question: "Merge Once, Validate, Hand to LightningCLI"

### What the pattern requires:

1. Read YAML chain (base -> stage -> scale -> model) -> deep merge
2. Apply dotted-key overrides (`trainer.max_epochs=2`)
3. Validate cross-field constraints (without importing torch/Lightning)
4. Emit single resolved dict preserving `class_path`/`init_args` structure
5. Hand to LightningCLI or write as YAML for SLURM job consumption

### Dynaconf assessment:

Fails at step 1. First-level-only YAML merge means `model.init_args` from a scale overlay
would replace the entire `model` dict from the stage YAML, losing `class_path`. This is a
fundamental architectural mismatch -- Dynaconf was designed for flat settings with environment
layering, not deeply nested ML config composition.

### OmegaConf assessment:

Handles steps 1-4 natively:
```python
from omegaconf import OmegaConf, DictConfig

def resolve_config(yaml_chain: list[Path], overrides: list[str]) -> dict:
    """Merge YAML chain + overrides, validate, return plain dict."""
    configs = [OmegaConf.load(f) for f in yaml_chain]
    if overrides:
        configs.append(OmegaConf.from_dotlist(overrides))
    merged = OmegaConf.merge(*configs)
    # Optional: merge with structured schema for type validation
    # schema = OmegaConf.structured(ConfigSchema)
    # merged = OmegaConf.merge(schema, merged)
    return OmegaConf.to_container(merged, resolve=True)
```

Step 5 is `yaml.dump(resolved)` or passing the dict to LightningCLI programmatically.

### Naive YAML merge assessment:

The current approach (`yaml.safe_load` + custom `deep_merge`) handles steps 1, 4, 5.
Step 2 requires `_flatten_dict()` (already implemented in `recipe_expand.py`).
Step 3 uses Pydantic `TrainingRunConfig` post-merge.

This is ~30 lines of code with zero dependencies. It works. The question is whether OmegaConf's
additions justify the dependency.

---

## 5. What OmegaConf Adds Over Naive Merge

| Capability | Naive merge | OmegaConf | Value for KD-GAT |
|---|---|---|---|
| Deep merge | Custom function | Built-in, tested | Low -- ours works |
| Dotlist overrides | Custom `_flatten_dict` | `from_dotlist()` native | Low -- ours works |
| Merge-time type validation | Post-merge Pydantic | Structured config during merge | Medium -- catches errors earlier |
| Interpolation (`${trainer.max_epochs}`) | Not available | Built-in resolvers | Low -- not currently needed |
| List merge modes | Not available | REPLACE/EXTEND/EXTEND_UNIQUE | Medium -- solves callback list issue |
| `???` (MISSING) sentinel | Not available | Built-in | Low -- Pydantic required fields serve this |
| Struct flag (reject unknown keys) | Not available | Built-in per-node | Low -- `extra="forbid"` on Pydantic |

The strongest argument for OmegaConf is **list merge control** (`ListMergeMode.EXTEND`).
The callback list replacement problem (`config_system_synthesis.md` Part 3) was solved via
`add_lightning_class_args` (forced callbacks in separate namespaces), but future cases could
benefit from `EXTEND` mode. However, this is a narrow win.

---

## 6. What OmegaConf Costs

1. **New dependency** -- `omegaconf` + `antlr4-python3-runtime`. Not heavy, but nonzero.
   Already in the PyG/Hydra ecosystem so may be transitively present.

2. **DictConfig impedance** -- OmegaConf's `DictConfig` is not a plain dict. Code that
   does `isinstance(x, dict)` fails. Must call `OmegaConf.to_container()` at the boundary.
   This is the exact "unpacking issue" noted in `config_system_synthesis.md` Part 1:
   "When a DictConfig is passed to code expecting a plain dict, the impedance mismatch surfaces."

3. **Lazy interpolation complexity** -- OmegaConf's `${key}` interpolation is lazy by default.
   Values aren't resolved until access time. This means a config that looks valid at merge time
   can fail at access time if a referenced key is missing. `OmegaConf.resolve()` forces eager
   resolution, but you have to remember to call it.

4. **Learning curve** -- `MISSING`, `DictConfig` vs `dict`, `struct` flag, `ListMergeMode`,
   resolver registration. None of these are hard, but they're concepts a contributor must learn.

5. **Two merge systems** -- If OmegaConf handles the dagster path's merge and jsonargparse
   handles the dev/test CLI path's merge, there are two merge implementations with subtly
   different semantics (OmegaConf list merge modes vs jsonargparse's atomic list replacement).
   This is the anti-pattern warned against in `config_system_synthesis.md` Part 6 point 3.

---

## 7. Recommendations

### Dynaconf: REJECT

**Reason:** First-level-only YAML merge is a fundamental blocker for nested ML configs.
Validation requires manual rule definitions (weaker than Pydantic). No CLI override support.
No ML ecosystem presence. Designed for a different problem domain (web app settings with
environment layering).

### OmegaConf: DO NOT ADOPT (investigate only if interpolation becomes needed)

**Reason:** OmegaConf does the merge correctly, but the naive approach already works and the
additions don't justify the costs for KD-GAT's specific situation:

1. **Deep merge** -- already implemented in ~15 lines. Battle-tested through recipe expansion
   and config validation (`validate.py` parses every config chain on every commit).

2. **Dotlist overrides** -- `_flatten_dict()` + `dict.update` already handles this. OmegaConf's
   `from_dotlist` is cleaner but not enough to justify a dependency.

3. **Type validation** -- `TrainingRunConfig` (Pydantic) + jsonargparse `__init__` introspection
   already provides two validation layers. OmegaConf's structured configs would be a third
   system, not a replacement.

4. **DictConfig impedance** -- introducing `DictConfig` objects into a codebase that passes
   plain dicts everywhere creates a new class of bugs. The synthesis doc explicitly warns about
   this.

5. **Two merge paths** -- the dev/test CLI path uses jsonargparse's merge. Adding OmegaConf
   for the dagster path creates the dual-merge anti-pattern.

**When to reconsider:**
- If interpolation (`${model.init_args.latent_dim}` referenced from multiple places) becomes
  a real need, OmegaConf's resolver system is the right tool.
- If the project adopts Hydra for sweep management (currently rejected, see synthesis doc Part 6).
- If list merge control becomes needed beyond the forced-callbacks solution.

### Recommended approach: Keep the naive YAML merge

The `ConfigResolver` from `config_system_synthesis.md` Part 5 should be implemented with:

```python
import yaml
from pathlib import Path
from graphids.config.contracts import TrainingRunConfig

def deep_merge(base: dict, override: dict) -> dict:
    """Recursive deep merge. Override wins. Lists replace atomically."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def resolve_config(
    yaml_chain: list[Path],
    overrides: dict[str, str] | None = None,
) -> dict:
    """Merge YAML chain, apply overrides, return plain dict."""
    configs = [yaml.safe_load(f.read_text()) for f in yaml_chain]
    merged = configs[0]
    for cfg in configs[1:]:
        merged = deep_merge(merged, cfg)
    if overrides:
        for dotted_key, value in overrides.items():
            parts = dotted_key.split(".")
            target = merged
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
    return merged
```

This is ~25 lines, zero dependencies, handles all current needs, and avoids the DictConfig
impedance and dual-merge-system risks. Pydantic `TrainingRunConfig` validates the boundary
fields post-merge. jsonargparse validates `__init__` signatures when LightningCLI consumes
the resolved config.

**The 30-line approach wins because the problem is simple.** Merging 3-4 YAML files with
dotted-key overrides is not a problem that needs a framework. The complexity in KD-GAT's
config system is in the *structure* (which files exist, what axes they represent, how they
compose) -- not in the *merge mechanics*. No library fixes structural problems.

---

## Sources

### Dynaconf
- [Home](https://www.dynaconf.com/)
- [Settings files](https://www.dynaconf.com/settings_files/)
- [Merging](https://www.dynaconf.com/merging/)
- [Validation](https://www.dynaconf.com/validation/)
- [GitHub](https://github.com/dynaconf/dynaconf) (v3.2.13, 4.3k stars)

### OmegaConf
- [GitHub](https://github.com/omry/omegaconf) (v2.3.0 stable, 1.9k stars)
- [Usage docs](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html)
- [Structured configs](https://omegaconf.readthedocs.io/en/latest/structured_config.html)
- [API reference](https://omegaconf.readthedocs.io/en/2.3_branch/api_reference.html)
- [Lightning-AI/pytorch-lightning#15109](https://github.com/Lightning-AI/pytorch-lightning/issues/15109) (OmegaConf + LightningCLI coexistence)
- [Merge precedence issue #1184](https://github.com/omry/omegaconf/issues/1184)
- [Deep merge issue #1080](https://github.com/omry/omegaconf/issues/1080)

### Project Context
- `plans/research/config_system_synthesis.md` -- canonical config architecture reference
- `graphids/config/contracts.py` -- `TrainingRunConfig` Pydantic model
- `graphids/config/recipe_expand.py` -- `_flatten_dict()` helper
- `graphids/orchestrate/validate.py` -- config chain validation
- `graphids/cli.py` -- `GraphIDSCLI`, `CLI_KWARGS`

### General
- [Dagster config docs](https://docs.dagster.io/concepts/configuration/config-schema)
- [Python Configuration Management survey](https://safjan.com/python-configuration-management/)
- [Kedro OmegaConf assessment](https://github.com/kedro-org/kedro/issues/1657)
