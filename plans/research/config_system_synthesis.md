# Config System Synthesis: KD-GAT

> Consolidated from two independent research threads: a pattern survey across 15+
> production systems, and an architectural analysis of config system tradeoffs for
> LightningCLI + Dagster stacks. This document is the canonical reference.
>
> Sources: `config_research.md` (pattern survey), `config_system_design.md` (architecture analysis)

---

## Part 1: The Config System Landscape

### Core Architectural Axes

Every config system sits somewhere on two axes:

- **Schema-first vs. composition-first** — does the schema live in code (type hints, Pydantic
  models) or in the config format itself (Jsonnet mixins, Dhall types, Hydra config groups)?
- **Static vs. dynamic resolution** — is the config fully resolved at parse time, or does it
  contain lazy references (`${interpolation}`) resolved at access time?

These axes explain the design choices — and failure modes — of every tool in the ecosystem.

### System Reference

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

**On OmegaConf / Hydra specifically:** OmegaConf's interpolation is lazy and struct-mode
enforcement is inconsistent. When two configs with overlapping keys are merged, or a `DictConfig`
is passed to code expecting a plain dict, the impedance mismatch surfaces. The `_target_` key
pattern for object instantiation is elegant in theory but produces deeply nested, hard-to-debug
configs in practice. The unpacking issues are structural, not incidental.

---

## Part 2: Three Patterns for Taming Combinatorial Explosion

Research across 15+ production systems surfaces three distinct patterns. Understanding which
pattern you're using — and its known failure modes — explains most config-related bugs.

### Pattern 1: Hierarchical Composition (Defaults Lists)

**Mechanism:** A primary config names which option from each axis to compose. Each axis is a
directory; each option is one small file. The framework merges them. CLI overrides win.

**File count scales linearly** with options per axis (N), not multiplicatively (N^K). A new axis
is a new directory. A new option is one file in that directory.

**Production evidence:**
- **Hydra** (Meta) — the canonical implementation. `--multirun model=a,b dataset=x,y` generates
  the cross product automatically.
  ([defaults list](https://hydra.cc/docs/advanced/defaults_list/),
  [experiment pattern](https://hydra.cc/docs/patterns/configuring_experiments/))
- **Habitat Lab** (Meta) — embodied AI stack with tasks x environments x agents x sensors x
  datasets. Defaults list reads like an assembly manifest.
  ([config README](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/README.md))
- **Fairseq** (Meta) — config groups mirror component types (`model/`, `task/`, `criterion/`).
  Registry pattern pairs each component with a dataclass.
  ([hydra integration](https://github.com/facebookresearch/fairseq/blob/main/docs/hydra_integration.md))
- **NeMo** (NVIDIA) — OmegaConf + Hydra + Fiddle under the hood.
  ([docs](https://docs.nvidia.com/nemotron/nightly/nemo_runspec/omegaconf.html))

**Known failure modes:** Breaks when axes aren't truly independent, or when you need structural
variation (not just value swaps between equivalent components).

---

### Pattern 2: Base + Overlay / Delta Patches

**Mechanism:** A complete, valid base config is the starting point. Thin overlays specify only
deltas. Deep-merge applies them. Max practical inheritance depth is ~3 levels.

**Per-variant cost is tiny** because overlays contain only deltas. MMDetection's 826
model-specific configs are thin because they inherit almost everything from 46 base files.

**Production evidence:**
- **MMDetection** (OpenMMLab) — 872 config files managed via 3-level `_base_` inheritance.
  46 base files, 826 model-specific configs. A ResNet-101 variant is ~3 lines changing backbone
  depth. `_delete_=True` handles structural swaps.
  ([config docs](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html))
- **Kustomize** (Kubernetes) — strategic merge patches that understand array semantics (merge by
  `name` field, not replace). Invented specifically because naive list replacement breaks real
  systems.
  ([docs](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/),
  [tutorial](https://glasskube.dev/blog/patching-with-kustomize/))
- **Helm** — `values.yaml` defaults + user override files per environment.
  ([values files](https://helm.sh/docs/chart_template_guide/values_files/))
- **Terraform** — generic modules + environment-specific root modules + workspaces.
  ([module composition](https://developer.hashicorp.com/terraform/language/modules/develop/composition))

**Known failure modes:**

**List replacement is the #1 trap.** Naive deep-merge replaces lists atomically — a stage YAML's
`callbacks:` list drops everything in the base and substitutes its own. This is not a bug in your
code; it's the expected behavior of YAML merge semantics. Kustomize invented strategic merge
patches specifically to solve this.

Base structural changes also cascade to all descendants — changing a field name in a base config
breaks every overlay that references it.

---

### Pattern 3: Programmatic Config (Code-as-Config with Deferred Instantiation)

**Mechanism:** Config is written in a real language (Python, Jsonnet, CUE). Files define data
structures describing what to build. A separate `instantiate()` / `build()` step creates live
objects. Full language power (functions, conditionals, imports) is available while keeping config
separate from execution.

**File growth is sub-linear** — functions generate configs parametrically. New variation
dimensions can be added without restructuring.

**Production evidence:**
- **Detectron2 LazyConfig** (Meta) — Python configs with `LazyCall` dicts + recursive
  `instantiate()`. Evolved from YACS because "YACS does not offer enough flexibility."
  ([tutorial](https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/lazyconfigs.md))
- **Fiddle** (Google/DeepMind) — `fdl.Config` wraps callable + args, `fdl.build()` instantiates
  recursively with memoization. Used by NeMo Run.
  ([repo](https://github.com/google/fiddle),
  [NeMo integration](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/configuration.html))
- **Jsonnet** (Databricks, Grafana) — Databricks runs 40K+ lines of Jsonnet across 1K+ files.
  Parametric construction (`newShard(name, env)`) replaces copy-paste entirely. Grafana's
  monitoring mixins compose dashboards and alerts as Jsonnet objects.
  ([Databricks blog](https://medium.com/databricks-engineering/declarative-infrastructure-with-the-jsonnet-templating-language-e33d97e862fd),
  [Tanka](https://grafana.com/blog/2020/03/11/how-the-jsonnet-based-project-tanka-improves-kubernetes-usage/))
- **CUE** (Istio, Dagger, Mercari) — constraints and values on a single continuum. Merge is
  commutative and idempotent; order never matters.
  ([docs](https://cuelang.org/docs/concept/configuration-use-case/),
  [Mercari](https://engineering.mercari.com/en/blog/entry/20220127-kubernetes-configuration-management-with-cue/))

**Known failure modes:** Config files become code — harder to review, higher learning curve,
debugging requires understanding the config language. A junior researcher editing a `.jsonnet`
file faces a steeper onramp than editing a `.yaml` file.

---

### Pattern Comparison

| Dimension | Composition | Base + Overlay | Code-as-Config |
|---|---|---|---|
| File growth | Linear per axis | Linear per variant | Sub-linear (parametric) |
| Adding a new axis | New directory | Restructure base | New function parameter |
| Traceability | Explicit (defaults list) | Implicit (merge order) | Traceable (imports/calls) |
| Validation | Structured configs | Runtime errors | CUE native; others ad-hoc |
| Best for | Moderate combinatorics | Environment variants | Rapid research iteration |
| Learning curve | Medium | Low | High |

---

## Part 3: Diagnosis of the KD-GAT Config System

### Current System Mapping

The current system is an **accidental hybrid of Patterns 1 and 2**, hitting the known failure
modes of both:

| Component | Pattern | Failure mode | Severity |
|---|---|---|---|
| `trainer.yaml` -> stage YAML -> overlay YAML | Pattern 2 (base + overlay) | **List replacement** — stage YAML's `callbacks:` atomically drops ModelCheckpoint. Already caused data loss (curriculum, 300 epochs, no checkpoint). | Critical |
| Overlay files (`small_gat.yaml`, etc.) | Pattern 1 (manual) | **Two axes in one file** — scale x model encoded together, forcing manual enumeration of cross product | Structural |
| `pipeline.yaml` + `resources.yaml` as topology | Neither — parallel declaration | **Drift** — `dgi/large` has no resource profile; `evaluation` stage has none; `medium` scale is dead config. Only caught at dagster load time. | Moderate |
| Recipe YAMLs enumerating configs | Manual enumeration | Doesn't scale — each new ablation dimension multiplies recipe entries | Moderate |
| Import-time YAML loading (`config/__init__.py`) | None (bare `yaml.safe_load`) | **No error handling** — malformed YAML crashes every import with unhelpful traceback | Low-moderate |
| `write_paths.yaml` + `run_dir()` | Duplicate declaration | **Template in YAML, f-string in Python** — change one, forget the other | Low |

### The Cross-Product Encoding Problem

The variation axes — `model_type`, `scale`, `stage`, `dataset` — are genuinely independent. This
is the ideal case for Pattern 1. But overlay files like `small_gat.yaml` and `large_vgae.yaml`
encode **two axes (scale x model) in a single file**, forcing manual enumeration of the cross
product. With 3 model types and 3 scales, that's 9 files instead of 6. With 4 model types and 4
scales, it's 16 files instead of 8. The explosion is quadratic in the number of axes encoded
together.

Separating these into independent config groups — one file per scale option, one file per
model-type option — makes the system compose naturally:

```
graphids/config/
  scales/
    small.yaml       # hidden_dim, latent_dim, num_layers — nothing model-specific
    large.yaml
  models/
    gat.yaml         # model_type-specific params only
    vgae.yaml
  stages/            # (unchanged — already independent)
    autoencoder.yaml
    normal.yaml
    ...
```

A run becomes:

```bash
python -m graphids fit \
    --config graphids/config/stages/autoencoder.yaml \
    --config graphids/config/scales/small.yaml \
    --config graphids/config/models/vgae.yaml
```

Each new model variant is one new file. Each new scale option is one new file. No cross product.
jsonargparse already supports multiple `--config` flags that compose left-to-right — this is
structurally equivalent to Hydra's defaults list without adopting Hydra.

### The Three Config Domains

The deeper structural problem is that KD-GAT has three config domains with no enforced contract
between them:

1. **Model/trainer config** — owned by LightningCLI, expressed as YAML mirroring class
   hierarchies, validated against dataclass schemas.
2. **Orchestration config** — owned by Dagster, expressed as `Config` subclasses, validated at
   job submission.
3. **Experiment config** — the sweep/ablation layer that neither system owns cleanly, accumulating
   as ad hoc YAML files and recipe enumerations.

This creates two additional failure modes beyond list replacement:

**Pack/unpack impedance** — two forms. *Vertical*: a config object serialized to YAML to cross a
process boundary (Dagster -> SLURM job -> `python -m graphids fit`) then deserialized back.
jsonargparse's `class_path`/`init_args` structure doesn't round-trip cleanly through a plain
dict. *Horizontal*: the same parameter (`batch_size`, `num_workers`) living in both `DataModule`
init args and `DagsterRunConfig` with no enforced equivalence between them.

**Write boundary drift** — `write_paths.yaml` defines where artifacts and logs go, but nothing
enforces that code actually uses those paths. Checkpoints, logs, and artifacts can end up
scattered across the filesystem when code bypasses the config definition.

---

## Part 4: Architecture Decision — Hybrid Approach

### Why Not Full Pydantic Consolidation

A full Pydantic consolidation — replacing LightningCLI's config system with a unified Pydantic
model hierarchy — would gain a single validation layer and explicit inter-system contracts but
lose:

- **LightningCLI's automatic CLI generation** — `--trainer.max_epochs`, `--model.lr`, flag
  generation from class signatures. This is non-trivial to replicate and valuable for dev/test
  invocations.
- **The `class_path`/`init_args` pattern** — specifying which concrete class to instantiate in
  YAML without changing code. Replacing this requires discriminated unions (enumerate every
  class upfront) or dynamic import validators.
- **jsonargparse's multi-source merge semantics** — the "base -> experiment -> CLI" composition
  model is built in. With Pydantic, you own that logic.

Given the primary usage pattern — large experiment pipelines through Dagster, occasional dev/test
CLI invocations — full consolidation trades too much ergonomics for the gain. The hybrid
approach keeps both systems and adds a typed contract between them.

### The Two-Level Fix

These operate at different levels and are not contradictory. Both are needed.

**Level 1: YAML restructuring (no new code)**

Fixes the file-level complexity and eliminates the combinatorial explosion.

1. Separate `scale x model` overlays into independent config axes (one file per axis option).
2. Fix list replacement: register critical callbacks via `add_lightning_class_args` so they
   live in separate namespaces immune to `trainer.callbacks:` list replacement.
3. Reorganize the directory structure to mirror the independent axes.

This can be done incrementally. It reduces file count, makes the composition model explicit, and
eliminates the manual cross-product enumeration in recipe YAMLs.

**Level 2: Narrow typed contract (~150 lines of new code)**

Fixes the system-boundary complexity between Dagster and Lightning.

A narrow `TrainingRunConfig` Pydantic model — covering only parameters that cross the
Dagster<->Lightning boundary or are actively swept — acts as the schema contract. The key scope
discipline: **this model should not mirror every `__init__` signature**. Internal Lightning
implementation details (layer widths, activation functions that are never swept) stay in
YAML/LightningCLI. Typically 10-20 parameters, not hundreds.

```
ExperimentConfig (Pydantic, narrow)     <- schema contract, not a mirror of everything
        |                    |
        v                    v
Dagster pipeline          LightningCLI / YAML
(programmatic)            (dev/test entry point)
```

---

## Part 5: Concrete Design

> **Implementation status (2026-04-01):** An interim field-passthrough override flow was built
> (recipe `trainer_overrides`/`resource_overrides` → `StageConfig` → two merge sites in
> `execution.py` and `assets.py`). This solves the immediate need (smoke test walltimes) but
> lacks cross-field validation and audit logging. **Decision: ConfigResolver remains the target
> architecture.** The interim flow will be subsumed — its helpers (`_flatten_dict`,
> `apply_resource_overrides`) become internal to the resolver, and the two merge sites collapse
> into one. See `issues/config-system-overhaul.md` P2.2 for migration path.

### Override Resolution

A single `ConfigResolver` is the **exclusive merge path for pipeline runs**. It does not layer
on top of jsonargparse's merge — it replaces it for that entry point. jsonargparse's native merge
is only used for dev/test CLI invocations. Having both active simultaneously recreates the
implicit resolution order problem you're trying to fix.

Override priority order (lowest -> highest):

1. Base YAML (`trainer.yaml`)
2. Axis config files (stage, scale, model)
3. Experiment overrides (Dagster-sourced)
4. Environment overrides (path remapping, resource limits per cluster)
5. CLI overrides (dev/test only — highest priority)

Validation happens on the **final merged state**, not on each source independently. This catches
cross-field constraint violations that partial validation misses (two individually valid configs
that produce an invalid combination).

```python
class ConfigResolver:
    def resolve(
        self,
        base_config_path: Path,
        axis_configs: list[Path],        # stage/, scale/, model/ files
        experiment_overrides: dict,      # from Dagster op config
        env_overrides: dict | None,      # path remapping, resource limits
        cli_overrides: dict | None,      # dev/test only
    ) -> TrainingRunConfig:
        sources = [base_config_path] + axis_configs
        base = deep_merge(*[yaml.safe_load(p.read_text()) for p in sources])
        resolved = deep_merge(base, experiment_overrides, env_overrides or {}, cli_overrides or {})
        return TrainingRunConfig.model_validate(resolved)  # validates final merged state
```

### Serialization Boundary

`TrainingRunConfig.to_lightning_yaml()` is the **only place** that knows about LightningCLI's
`class_path`/`init_args` conventions. This is a known fragility — any Lightning upgrade that
changes the YAML format breaks this method. The mitigation is keeping the model narrow: a
`to_lightning_yaml()` covering 10-20 fields has a small blast radius. If the model grows to
mirror hundreds of fields, this becomes a serious maintenance liability.

```python
class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")  # catches schema drift at definition time

    def to_lightning_yaml(self) -> dict:
        """Only place that knows about LightningCLI's YAML conventions."""
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
```

Dagster holds a pointer to the schema plus a small delta — never a copy of the full schema:

```python
class DagsterRunConfig(Config):
    experiment_name: str
    base_config: str         # path to base YAML — not a copy of its contents
    axis_configs: list[str]  # e.g. ["graphids/config/scales/small.yaml", "graphids/config/models/gat.yaml"]
    overrides_json: str = "{}"
```

### Write Boundary Enforcement

`PathContext` is an immutable object that is the **only source of valid write paths**.
`frozen=True` prevents mutation after construction. All paths are computed properties — there is
no free string to accidentally misuse. Code that wants to write checkpoints, logs, or artifacts
must accept a `PathContext` and use its properties.

`write_paths.yaml` already tries to be this registry. The problem it has is enforcement — code
can read the YAML and then write somewhere else anyway. A frozen Pydantic model with computed
properties cannot be accidentally bypassed in the same way.

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
        """Resolved config written before training — the reproducibility artifact."""
        return self.artifacts_dir / "config.yaml"

    def ensure_dirs(self) -> None:
        for d in [self.checkpoints_dir, self.logs_dir, self.artifacts_dir]:
            d.mkdir(parents=True, exist_ok=True)
```

On the Dagster side, `PathContext` is a `ConfigurableResource` constructed from env vars and
injected into every op that writes anything. The `base_dir` points to the shared data directory
in pipeline mode and a local directory in dev/test mode — behavior is identical, only paths
differ.

### The Central Invariant

> The resolved `TrainingRunConfig` is always written to `PathContext.config_snapshot_path`
> before training starts.

This one file is the complete reproducibility artifact. Any run can be re-executed with:

```bash
python -m graphids fit --config artifacts/<experiment>/<run_id>/config.yaml
```

regardless of which entry point originally produced it.

### Full Pipeline Flow

```
Dagster Launchpad
    |  DagsterRunConfig(experiment_name, base_config, axis_configs[], overrides)
    v
ConfigResolver.resolve()
    |  merges: base -> axis configs -> Dagster overrides -> env overrides
    |  validates final merged state
    |  emits override audit log (which override came from which source)
    v
TrainingRunConfig  (validated Pydantic object, narrow schema)
    |
    |-- .to_lightning_yaml()
    |       written to PathContext.config_snapshot_path before training starts
    |       this file is the complete reproducibility artifact
    |
    +-- PathContext constructed from TrainingRunConfig.paths
            injected into LightningCLI callbacks and Dagster ops
            enforces all writes go to shared data directory
```

For dev/test:

```bash
python -m graphids fit \
  --config graphids/config/stages/autoencoder.yaml \
  --config graphids/config/scales/small.yaml \
  --config graphids/config/models/vgae.yaml \
  --trainer.max_epochs=5
```

Same `PathContext` enforcement, `base_dir` from a local env var. The `config_snapshot_path` is
still written — a dev run is also reproducible.

---

## Part 6: What NOT to Do

Both research threads converge on these anti-patterns:

1. **Don't adopt Hydra.** jsonargparse's multi-`--config` composition is equivalent to Hydra's
   defaults list for independent axes. Switching frameworks buys lazy interpolation (not needed)
   and loses tight Lightning integration (very needed). OmegaConf's lazy resolution reintroduces
   the unpacking issues already documented.

2. **Don't build a full Pydantic mirror of every `__init__` signature.** `TrainingRunConfig` is
   a boundary contract (10-20 swept/cross-boundary params), not a complete config schema.
   Internal model details (layer widths, activation functions never swept) stay in YAML where
   jsonargparse validates them against type annotations.

3. **Don't add a parallel merge path.** If `ConfigResolver` is introduced, it must *replace*
   jsonargparse's merging for the dagster path, not run alongside it. Two merge paths = two
   sources of truth = drift.

4. **Don't template YAML.** Jsonnet/Jinja templating is Pattern 3 — powerful but adds a
   language. The combinatorial explosion is solvable with independent axes (Pattern 1) within
   plain YAML + jsonargparse.

5. **Don't maintain parallel topology declarations.** If `pipeline.yaml` declares stages and
   models, `resources.yaml` must not independently enumerate them. Either merge them or add an
   import-time cross-validation assertion (like the existing `ckpt_stages` check).

---

## Part 7: Implementation Order

### Phase 1 — YAML restructuring + forced callbacks (no new abstractions)

| Action | Fixes | Risk | Priority |
|---|---|---|---|
| **Forced callbacks** via `parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")` and `add_lightning_class_args(EarlyStopping, "early_stopping")`. Stage YAMLs override via `checkpoint.monitor: val_acc`, not `trainer.callbacks: [...]`. Plan ready: `plans/architecture/forced-callbacks.md`. | Data loss from silent callback drop | None — strictly additive | P0 |
| **Separate cross-product overlays** into independent axes. Split `small_gat.yaml` -> `scales/small.yaml` + `models/gat.yaml`. Update `--config` invocations to compose them. | Manual cross-product enumeration, missing overlay silent skip | Low — file reorganization | P1 |
| **Import-time cross-validation** of `resources.yaml` vs `pipeline.yaml`. Extend the `ckpt_stages` assertion pattern. | Drift between topology and resource profiles | None — one assertion | P1 |
| **Reorganize directory structure** to mirror independent axes (`stages/`, `scales/`, `models/`). Update recipe YAMLs to reference axis files instead of cross-product files. | Directory structure reflects composition model | Low — update references | P1 |

**Measurable outcome:** File count stops growing multiplicatively when new model types or scales
are added. Recipe YAMLs become short lists of axis selections, not enumerations of full configs.
No training run can silently lose checkpoints.

### Phase 2 — Narrow typed contract (~150 lines, Dagster<->Lightning boundary)

| Action | Fixes | Risk | Priority |
|---|---|---|---|
| Define `TrainingRunConfig` covering only boundary/swept parameters. Start narrow — add fields only when needed. | Pack/unpack impedance, dagster can't validate training config | Medium — scope discipline required | P2 |
| Implement `to_lightning_yaml()` / `from_lightning_yaml()` as single serialization boundary. | Multiple serialization formats | Medium — must track Lightning YAML conventions | P2 |
| Implement `ConfigResolver` as exclusive merge path for pipeline runs. | Implicit override resolution order | Medium — replaces existing merge path | P2 |
| Replace `write_paths.yaml` with frozen `PathContext`. Update callbacks and ops to use its properties. | Duplicate path declarations, unenforced write boundaries | Medium — touches all write sites | P2 |

**Measurable outcome:** A Dagster run produces a `config.yaml` snapshot that fully reproduces the
run. Write paths are no longer scattered — every artifact lands in the expected location.

### Phase 3 — Ongoing scope discipline + optional enhancements

| Action | Fixes | Risk | Priority |
|---|---|---|---|
| Treat any `TrainingRunConfig` field addition as a deliberate decision. `extra="forbid"` catches accidental additions. | Model bloat | Ongoing vigilance | P3 |
| Recipe generation as code (Pattern 3 for sweep enumeration only). | Recipe YAML doesn't scale with ablation dimensions | Low — isolated to orchestrate/ | P3 |

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Full Pydantic consolidation | No | Loses LightningCLI CLI generation and `class_path` dispatch; too high ergonomics cost |
| Hydra for sweep layer | No | OmegaConf's lazy resolution creates unpacking issues; adding it back solves one problem, reintroduces another |
| Jsonnet as config generator | Optional (P3) | Valid for sweep generation; generates YAMLs that LightningCLI reads, no runtime coupling |
| YAML axis restructuring | Yes (Phase 1) | Zero new code, directly fixes combinatorial explosion, immediately reduces file count |
| Forced callbacks via `add_lightning_class_args` | Yes (Phase 1, P0) | Verified fix per `forced-callbacks.md`; separate namespaces immune to list replacement |
| Import-time topology validation | Yes (Phase 1) | Extends existing `ckpt_stages` assertion pattern to resource profiles |
| Narrow `TrainingRunConfig` | Yes (Phase 2) | Fixes pack/unpack impedance; scope discipline is the maintenance risk |
| `PathContext` enforcement | Yes (Phase 2) | Replaces `write_paths.yaml` with actual enforcement; frozen model cannot be bypassed |
| `ConfigResolver` as exclusive pipeline merge | Yes (Phase 2) | Must replace jsonargparse's merge for pipeline runs, not layer on top of it |

---

## References

### Pattern Survey Sources

- Hydra: [defaults list](https://hydra.cc/docs/advanced/defaults_list/), [experiments](https://hydra.cc/docs/patterns/configuring_experiments/)
- Habitat Lab: [config README](https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/config/README.md)
- Fairseq: [hydra integration](https://github.com/facebookresearch/fairseq/blob/main/docs/hydra_integration.md)
- NeMo: [OmegaConf config](https://docs.nvidia.com/nemotron/nightly/nemo_runspec/omegaconf.html), [NeMo Run CLI](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemorun/guides/cli.html)
- MMDetection: [config docs (3.x)](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/config.html), [config docs (2.17)](https://mmdetection.readthedocs.io/en/v2.17.0/tutorials/config.html)
- Kustomize: [docs](https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/), [tutorial](https://glasskube.dev/blog/patching-with-kustomize/), [vs Helm (IBM)](https://www.ibm.com/think/insights/kustomize-vs-helm)
- Helm: [values files](https://helm.sh/docs/chart_template_guide/values_files/), [subcharts](https://helm.sh/docs/chart_template_guide/subcharts_and_globals/)
- Terraform: [module composition](https://developer.hashicorp.com/terraform/language/modules/develop/composition)
- Detectron2: [LazyConfig](https://github.com/facebookresearch/detectron2/blob/main/docs/tutorials/lazyconfigs.md), [config system (DeepWiki)](https://deepwiki.com/facebookresearch/detectron2/4.2-configuration-system)
- Fiddle: [repo](https://github.com/google/fiddle), [docs](https://fiddle.readthedocs.io/en/latest/)
- Jsonnet/Databricks: [blog](https://medium.com/databricks-engineering/declarative-infrastructure-with-the-jsonnet-templating-language-e33d97e862fd)
- Grafana/Tanka: [blog](https://grafana.com/blog/2020/03/11/how-the-jsonnet-based-project-tanka-improves-kubernetes-usage/), [jsonnet-libs](https://github.com/grafana/jsonnet-libs)
- CUE: [config use case](https://cuelang.org/docs/concept/configuration-use-case/), [Holos evaluation](https://holos.run/blog/why-cue-for-configuration/), [Mercari](https://engineering.mercari.com/en/blog/entry/20220127-kubernetes-configuration-management-with-cue/)
- Comparison: [Jsonnet vs Dhall vs CUE](https://pv.wtf/posts/taming-the-beast)

### Architecture Analysis Sources

- jsonargparse: [docs](https://jsonargparse.readthedocs.io/)
- OmegaConf: [docs](https://omegaconf.readthedocs.io/)
- Pydantic Settings: [docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- Dagster Config: [docs](https://docs.dagster.io/guides/build/configuring-ops-resources)
- GIN-Config: [repo](https://github.com/google/gin-config)
- Dhall: [docs](https://dhall-lang.org/)
- [10 Hydra/YAML Config Patterns](https://medium.com/@ThinkingLoop/10-hydra-yaml-config-patterns-that-keep-you-sane-04eed3d1c28f)
- [Techniques for Configurable Python Code](https://guillaumegenthial.github.io/config-python-techniques.html)
- [The Configuration Complexity Curse](https://blog.cedriccharly.com/post/20191109-the-configuration-complexity-curse/)
