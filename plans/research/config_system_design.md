# Configuration System Design: Cross-Analysis and Architecture

> Summary of a design discussion covering config system landscape, tradeoffs, and a hybrid
> architecture for a research ML stack using LightningCLI, jsonargparse, and Dagster.

---

## 1. The Config System Landscape

### Core Architectural Axes

Config systems split along two fundamental axes:

- **Schema-first vs. composition-first** — does the schema live in code (type hints, Pydantic models)
  or in the config format itself (Jsonnet mixins, Dhall types)?
- **Static vs. dynamic resolution** — is the config fully resolved at parse time, or does it
  contain lazy references (`${interpolation}`) resolved at access time?

Understanding these two axes explains most of the design choices — and failure modes — of every
tool in the ecosystem.

---

### System-by-System Analysis

#### OmegaConf / Hydra

**Architecture**: OmegaConf is a structured config library with a two-tier resolution model.
`DictConfig` and `ListConfig` objects behave like dicts/lists but defer `${path.to.key}`
interpolation until access time. Hydra adds config group composition, override syntax, and a
launcher abstraction on top.

**Design intent**: Solving the ML experiment management problem. The "compose config groups at
runtime" model directly targets "run 50 experiments varying optimizer and dataset independently."
The override syntax (`+trainer.lr=1e-3 ++model=resnet50`) is designed for cluster job submission.

**Why unpacking issues happen**: OmegaConf's interpolation is lazy and struct-mode enforcement is
inconsistent. When two configs with overlapping keys are merged, or a `DictConfig` is passed to
code expecting a plain dict, the impedance mismatch surfaces. The `_target_` key pattern for object
instantiation is elegant in theory but creates deeply nested, hard-to-debug configs in practice.

| | |
|---|---|
| **Strengths** | Multi-run sweep syntax, config group composition, callback/plugin system |
| **Weaknesses** | Lazy resolution creates subtle bugs, non-obvious merge semantics, poor IDE support for interpolation targets, config becomes a DSL |

---

#### jsonargparse + LightningCLI

**Architecture**: Schema-first. Reads Python type annotations and generates both CLI argument
parsing and YAML deserialization simultaneously. `class_from_function` and `lazy_instance` defer
object construction until all args are resolved, avoiding the circular dependency problem
Hydra's `instantiate()` has.

**Design intent**: Eliminate the boilerplate of writing argparse + config loading + validation
separately. LightningCLI uses `Trainer`, `LightningModule`, and `DataModule` as the schema —
changing a model changes the valid config keys automatically.

| | |
|---|---|
| **Strengths** | Automatic CLI generation, bidirectional YAML↔type resolution, tight Lightning integration |
| **Weaknesses** | Deep nesting mirrors class hierarchy awkwardly for non-Lightning code, cryptic type mismatch errors, `link_arguments` has subtle evaluation order issues |

---

#### Jsonnet

**Architecture**: A pure functional language that is a strict superset of JSON. Every `.jsonnet`
file is a function returning a JSON value. Composition is through imports, mixins (the `+:`
merge operator), and parameterized objects. Born at Google for managing k8s/GCP configs at scale.

**Core insight**: configs need inheritance, abstraction, and reuse — instead of bolting those onto
YAML as magic syntax, make a real language with a JSON output type.

```jsonnet
local base = { lr: 0.001, batch_size: 32 };
local big_model = base + { batch_size: 256 };
```

| | |
|---|---|
| **Strengths** | Fully composable, excellent for templating large config families, deterministic evaluation |
| **Weaknesses** | Another language to learn, sparse tooling outside k8s/infra, not integrated with Python type systems |

---

#### Dhall

**Architecture**: A total functional language — provably terminates (no infinite loops possible).
Has a type system with polymorphism. Non-Turing-completeness is intentional: it lets you
statically analyze what a config will produce. Imports support content-addressed caching
(`sha256hash:...` pins remote imports).

| | |
|---|---|
| **Strengths** | Type safety, guaranteed termination, mathematically analyzable |
| **Weaknesses** | Haskell-level learning curve, small community, Python bindings are not first-class |

---

#### Cue

**Architecture**: A constraint language where data and schemas are the same thing — you unify a
value against a type, and unification is the fundamental operation. Values live in a lattice.
Grew out of GCL (Google's internal config language).

**Design intent**: Unify configuration, data validation, and policy enforcement. The same `.cue`
file can validate a YAML file, generate a YAML file, or be the config itself.

| | |
|---|---|
| **Strengths** | Compose partial configs and constraints freely without worrying about merge order, `cue vet` validates any data format against schemas |
| **Weaknesses** | Values-as-constraints is non-intuitive, Python support is incomplete, small ML community |

---

#### Pydantic Settings

**Architecture**: Schema-first with pluggable sources — env vars, `.env` files, YAML/JSON/TOML
loaders, secrets managers, etc. Not a config format but a config *ingestion* framework.
`model_validator` and `field_validator` hooks enable complex cross-field validation.

| | |
|---|---|
| **Strengths** | Native Python, excellent error messages, env var override is first-class, integrates with FastAPI/Typer |
| **Weaknesses** | No built-in "merge N config files" model, no CLI generation |

---

#### Dagster Config

**Architecture**: Pydantic-based `Config` classes defined per Op, Asset, Resource, and Schedule.
The Dagster UI introspects and renders these as forms. Execution-time config is validated at job
submission before dispatching to workers.

**Design intent**: Config is tightly coupled to the execution graph — the schema lives with the
code. This is the right tradeoff for an orchestrator that must validate configs before dispatch,
show them in a UI, and log them for reproducibility.

---

#### GIN-Config

**Architecture**: Python-native dependency injection. Functions decorated with `@gin.configurable`
have their argument defaults set globally from a `.gin` file. Useful for modifying deeply nested
defaults (e.g., "use LayerNorm instead of BatchNorm in all ResBlocks") without threading config
through every call site.

| | |
|---|---|
| **Strengths** | Powerful for modifying deeply nested code, no boilerplate at call sites |
| **Weaknesses** | Global mutable state, hard to reason about in large codebases, not popular outside DeepMind-adjacent research |

---

### Comparison Matrix

| System | Schema source | Composition model | Type safety | CLI gen | Best fit |
|---|---|---|---|---|---|
| jsonargparse | Python type hints | YAML merge + CLI overrides | Strong | Yes | ML training configs |
| Hydra / OmegaConf | Optional (structured) | Config groups, lazy interpolation | Optional | Partial | Sweep experiments |
| Pydantic Settings | Pydantic models | Source priority chain | Strong | No | Service / app config |
| Dagster Config | Pydantic + `Config` class | Per-job, execution-time | Strong | No | Orchestration params |
| Jsonnet | None (JSON output) | Imports + mixins + functions | None | No | Templated config generation |
| Dhall | Built-in type system | `let..in` + imports | Very strong | No | Policy-enforced infra |
| Cue | Unified (value = schema) | Unification lattice | Strong, novel | No | Cross-tool validation |
| GIN-Config | Python decorators | Global scope injection | Weak | No | Research DI overrides |

---

## 2. The Problem with a Split Config Stack

### The Three Config Domains

A stack combining LightningCLI and Dagster has three config domains with no enforced contract
between them:

1. **Model/trainer config** — owned by LightningCLI, expressed as YAML mirroring class hierarchies,
   validated against dataclass schemas.
2. **Orchestration config** — owned by Dagster, expressed as `Config` subclasses, validated at job
   submission.
3. **Experiment config** — the "run this sweep" layer that neither system owns cleanly, accumulating
   as ad hoc YAML files.

### The Core Structural Tensions

**Override propagation** — multiple override sources (Dagster launch params, experiment YAML, CLI
flags, env vars) with no enforced resolution order. The order is implicit: whatever jsonargparse
merges last wins, with no audit trail. Cross-field constraint violations can pass individual-layer
validation but fail on the resolved combination.

**Pack/unpack impedance** — two forms:

- *Vertical*: a config object serialized to YAML to cross a process boundary (Dagster → subprocess)
  then deserialized back. jsonargparse's YAML structure (`class_path`/`init_args`) doesn't
  round-trip cleanly through a plain `.model_dump()`.
- *Horizontal*: a parameter living in one system's schema but needing visibility in another's.
  `batch_size` might be a `DataModule` init arg in Lightning's world and a `DagsterRunConfig` field
  in Dagster's — two representations of the same value with no enforced equivalence.

**Write boundary enforcement** — no enforced rule preventing code from writing artifacts and logs
to arbitrary paths. The write destination is implicit, split across logger config, callback config,
and environment assumptions.

---

## 3. The Pydantic Consolidation: Tradeoff Analysis

### What Full Consolidation Gains

- **Single validation layer** across all three domains — schema defined once, both systems validate
  against it. Type errors surface at the boundary, not inside training runs.
- **Explicit, inspectable contract** between Dagster and Lightning. The current implicit
  protocol (JSON blob passed to a subprocess) becomes a typed, loggable `TrainingRunConfig`.
- **Composable config construction in plain Python** — `model_copy(update={...})` is testable,
  introspectable, diffable. Sweep generation becomes a Python loop rather than Jsonnet or Hydra
  multirun.
- **IDE support** — autocomplete, go-to-definition, and rename refactoring work across the full
  config surface including inside Dagster ops.
- **Testable config logic** — unit tests can assert on config construction: "does this sweep
  correctly enumerate all combinations?"

### What Full Consolidation Loses

- **LightningCLI's automatic CLI generation** — the most significant cost. LightningCLI
  introspects `Trainer`, `LightningModule`, and `DataModule` signatures, generates
  `--trainer.max_epochs`-style flags, handles `class_path`/`init_args` for swapping
  implementations, and resolves `link_arguments`. Replacing this requires either re-implementing
  that introspection or giving up command-line overrides.
- **The `class_path`/`init_args` pattern for dynamic dispatch** — specifying which concrete
  class to instantiate in YAML. Reproducing this in Pydantic requires a discriminated union
  (enumerating every possible class upfront) or a custom validator doing dynamic imports.
- **jsonargparse's multi-source merge semantics** — the "base config → experiment config → CLI
  overrides" composition model is built in. With Pydantic you own that logic.
- **The Dagster UI config form** — Dagster renders `Config` subclasses as launch forms. A deeply
  nested Pydantic model would need a thin `DagsterRunConfig` surface anyway.

### Usage Pattern Determines the Right Answer

The question that resolves the tradeoff: **how often is training invoked from the command line
vs. always through Dagster?**

For a stack primarily running large experiment pipelines through Dagster, with occasional 1-off
dev/test runs, **full consolidation trades too much ergonomics for too little gain**. The hybrid
approach is the correct architecture.

---

## 4. The Hybrid Architecture

### Design Principle

The split between pipeline mode and dev/test mode is not a bug — it reflects a real boundary.
These two modes have genuinely different requirements:

- **Pipeline mode**: correctness, reproducibility, full validation, Dagster owns the entry point,
  config constructed programmatically.
- **Dev/test mode**: speed, ergonomics, CLI overrides, LightningCLI as the entry point.

The Pydantic model is not a replacement for either system — it is a **schema contract** that both
entry points serialize to and from.

```
ExperimentConfig (Pydantic)          ← single schema, source of truth
       │                    │
       ▼                    ▼
Dagster pipeline          LightningCLI / YAML
(programmatic)            (dev/test entry point)
```

---

### Layer 1: Explicit Override Resolution

Override resolution is made explicit and unidirectional through a single `ConfigResolver`.
Validation happens on the **final merged state**, not on each source independently — catching
cross-field constraint violations that partial validation misses. The resolver also emits a config
audit log recording which override came from which source.

```python
class ConfigResolver:
    def resolve(
        self,
        base_config_path: Path,
        experiment_overrides: dict,   # from Dagster op config
        env_overrides: dict | None,   # path remapping, resource limits
        cli_overrides: dict | None,   # only in dev/test mode, highest priority
    ) -> TrainingRunConfig:
        base = TrainingRunConfig.model_validate(
            yaml.safe_load(base_config_path.read_text())
        )
        resolved = deep_merge(base, experiment_overrides, env_overrides or {}, cli_overrides or {})
        return TrainingRunConfig.model_validate(resolved)  # validates final merged state
```

**Override priority order** (lowest → highest):
1. Base YAML
2. Experiment overrides (Dagster-sourced)
3. Environment overrides (path remapping)
4. CLI overrides (dev/test only)

---

### Layer 2: Serialization Boundary

`TrainingRunConfig` owns the serialization format. `to_lightning_yaml()` is the **only place**
that knows about LightningCLI's `class_path`/`init_args` conventions. If Lightning changes its
YAML format, one method changes.

```python
class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # catches schema drift immediately

    def to_lightning_yaml(self) -> dict:
        """The only place that knows about LightningCLI YAML conventions."""
        return {
            "trainer": self.trainer.model_dump(),
            "model": {
                "class_path": self.model.class_path,
                "init_args": self.model.init_args.model_dump(),
            },
            "data": {
                "class_path": self.data.class_path,
                "init_args": self.data.init_args.model_dump(),
            },
        }

    @classmethod
    def from_lightning_yaml(cls, d: dict) -> "TrainingRunConfig":
        """Inverse — reads an existing LightningCLI YAML back into the canonical form."""
        ...
```

Dagster holds a *pointer* to the schema plus a small delta — never a copy of the full schema:

```python
class DagsterRunConfig(Config):
    # Only fields a human would override at launch time
    experiment_name: str
    base_config: str        # path to base YAML, not a copy of its contents
    overrides_json: str = "{}"
```

---

### Layer 3: Write Boundary Enforcement

A `PathContext` object is constructed once and is the **only source of valid write paths**.
`frozen=True` prevents mutation after construction. All paths are computed properties — there is
no free string to accidentally misuse.

```python
class PathContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    experiment_name: str
    base_dir: Path

    @property
    def checkpoints_dir(self) -> Path:
        return self.base_dir / "checkpoints" / self.experiment_name / self.run_id

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs" / self.experiment_name / self.run_id

    @property
    def artifacts_dir(self) -> Path:
        return self.base_dir / "artifacts" / self.experiment_name / self.run_id

    @property
    def config_snapshot_path(self) -> Path:
        """Resolved config YAML written before training starts — the reproducibility artifact."""
        return self.artifacts_dir / "config.yaml"

    def ensure_dirs(self) -> None:
        for d in [self.checkpoints_dir, self.logs_dir, self.artifacts_dir]:
            d.mkdir(parents=True, exist_ok=True)
```

On the Dagster side, `PathContext` becomes a `ConfigurableResource` — constructed from env vars,
injected into every op that writes anything:

```python
class SharedDataResource(ConfigurableResource):
    base_dir: str

    def path_context_for(self, run_id: str, experiment_name: str) -> PathContext:
        return PathContext(
            run_id=run_id,
            experiment_name=experiment_name,
            base_dir=Path(self.base_dir),
        )
```

LightningCLI's loggers and checkpoint callbacks are overridden in `before_fit` to use paths
from `PathContext` — not the paths that came in from YAML:

```python
class PathAwareLightningCLI(LightningCLI):
    def before_fit(self):
        ctx = PathContext(
            run_id=self.config["run_id"],
            experiment_name=self.config["experiment_name"],
            base_dir=Path(self.config["base_dir"]),
        )
        ctx.ensure_dirs()
        self.trainer.logger.log_dir = str(ctx.logs_dir)
        self.trainer.checkpoint_callback.dirpath = str(ctx.checkpoints_dir)
        ctx.config_snapshot_path.write_text(yaml.dump(self.config.as_dict()))
```

---

### Full Pipeline Flow

```
Dagster Launchpad
    │  DagsterRunConfig(experiment_name, base_config path, small overrides)
    ▼
ConfigResolver.resolve()
    │  merges: base YAML → Dagster overrides → env overrides → (CLI overrides in dev)
    │  validates final merged state
    │  emits override audit log
    ▼
TrainingRunConfig  (validated Pydantic object)
    │
    ├── .to_lightning_yaml()  →  written to PathContext.artifacts_dir/config.yaml
    │                            (the only serialization format LightningCLI sees)
    │
    └── PathContext constructed from TrainingRunConfig.paths
            │
            └── injected into LightningCLI + Dagster ops
                enforces all writes go to shared data dir
```

For a dev/test run, the same `ConfigResolver` is used with CLI overrides as the highest-priority
source. `PathContext` enforces write boundaries — a local `base_dir` from an env var replaces the
shared storage path. Behavior is identical; only paths differ.

---

### The Central Invariant

> The resolved `TrainingRunConfig` is always written to `PathContext.config_snapshot_path`
> before training starts.

This one file is the complete reproducibility artifact. Any run can be re-executed with:

```bash
python train.py --config artifacts/<experiment>/<run_id>/config.yaml
```

regardless of which entry point originally produced it.

---

## 5. Scope Discipline

The risk of this architecture is `TrainingRunConfig` growing to mirror every LightningCLI
parameter, creating a maintenance burden where adding a new model field requires updating both the
LightningModule and the Pydantic model.

The mitigation is to scope `TrainingRunConfig` **narrowly** — only parameters that:
- cross the Dagster↔Lightning boundary, or
- are actively swept over in experiments.

Internal Lightning implementation details (layer widths, activation functions that are never
swept) stay in the YAML/LightningCLI layer. The Pydantic model is the **interface**, not a mirror
of the full config tree. Typically this means 10–20 parameters, not hundreds.

This boundary discipline is the main ongoing design decision. If the model stays narrow, the two
systems remain genuinely independent with a clean contract between them.
