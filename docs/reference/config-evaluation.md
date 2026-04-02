# CLI + Config System Evaluation

> Independent MLOps review of KD-GAT's config and CLI architecture
> Evaluated against: PyTorch Lightning/LightningCLI (jsonargparse), Hydra/OmegaConf (Meta), MMEngine (OpenMMLab), Dagster config
> Date: 2026-04-01

## Executive Summary

KD-GAT's config system is a well-architected solution for a research ML pipeline with an unusual constraint profile: multi-stage DAG execution across an HPC SLURM environment where torch cannot be imported at definition time. The design makes several non-obvious correct choices -- particularly the lazy torch import boundary, forced callbacks via parser namespaces, and the single `run_lightning()` convergence point. The main risks are the dual merge implementation (naive `deep_merge` vs jsonargparse's type-aware merge) and the recipe expansion system's growing complexity, both of which are tractable fixes rather than architectural flaws.

## Strengths

### S1. Clean torch import boundary (strongest architectural decision)

The separation of `cli.py` (torch-free, importable on login nodes and dagster workers) from `_lightning.py` (torch-dependent, lazy-imported) is the single most important design decision in the system. Evidence:

- `cli.py` imports only `yaml_utils` (`cli.py:14`), never torch or Lightning
- `run_lightning()` does the lazy import at call time (`cli.py:52-53`)
- `__main__.py` defers torch multiprocessing setup to `_run_lightning()` (`__main__.py:54-57`)

**Industry comparison:** Neither Hydra nor MMEngine address this concern at all -- they assume a single-process context. LightningCLI itself doesn't solve this (importing `LightningCLI` pulls in torch). Dagster's Component model is the closest parallel, but KD-GAT's approach is more explicit. This is a genuine innovation born from real HPC constraints.

### S2. Single convergence point for training

All paths (dev CLI, pipeline/dagster, validation) converge at `GraphIDSCLI(**CLI_KWARGS, args=args)` (`_lightning.py:36-54`). The architecture document makes this explicit as a key invariant (line 74: "Routes A and B converge at `run_lightning()` -> `GraphIDSCLI(LightningCLI)`").

**Why this matters:** Hydra-based systems often end up with separate code paths for interactive use vs orchestrated runs (one using `@hydra.main`, another using `compose` API). MMDetection's `train.py` vs `tools/test.py` are separate scripts. KD-GAT avoids this divergence entirely.

### S3. Forced callbacks via parser namespaces

The `add_lightning_class_args(ModelCheckpoint, "checkpoint")` pattern (`_lightning.py:44-51`) solves a real problem: YAML list replacement semantics dropping critical callbacks. This is documented in the architecture doc (lines 166-179) and the solution is correct -- by placing checkpoint/early_stopping in their own parser namespaces (`checkpoint.*`, `early_stopping.*`), they can't be clobbered by `trainer.callbacks: [...]` overrides.

**Industry comparison:** Hydra handles this with `defaults` list composition, but you can still accidentally override a list. LightningCLI's own docs don't recommend this pattern -- it's a project-specific invention. MMEngine uses a registry-based `default_hooks` dict (not a list) which avoids the problem differently.

### S4. Import-time config tree validation

`topology.py:90-114` validates that every `(model_type, scale)` combination has config files and resource profiles at import time. This is essentially a compile-time check -- if you add a model family to `axes.yaml` but forget the scale YAML, the system fails immediately on any import of `graphids.config`.

**Industry comparison:** Hydra discovers missing configs at runtime (when the config group is referenced). MMEngine's registry similarly validates at runtime. This project's approach catches drift earlier.

### S5. Pydantic contracts with `extra="forbid"`

`TrainingSpec` (`models.py:10-25`), `TrainingRunConfig` (`contracts.py:35-38`), `KDEntry` (`contracts.py:13-14`) all use `extra="forbid"`. This means typos in config fields are caught at construction time, not silently ignored. The `ContractEnvelope` pattern (`models.py:28-36`) adds versioning for the dagster-SLURM transport layer.

**Industry comparison:** Dagster's own `Config(BaseModel)` uses the same pattern. Hydra's structured configs with `@dataclass` catch extras via OmegaConf's strict mode but don't provide envelope versioning. This is well-implemented.

### S6. Identity hashing for reproducible run directories

`compute_identity_hash()` (`paths.py:117-141`) produces deterministic 8-char hashes from `identity_keys` defined per stage in `topology.py`. Missing keys raise `KeyError` (line 133) rather than silently hashing to defaults. The path layout `{lake_root}/dev/{user}/{dataset}/{model}_{scale}_{stage}_{hash}/seed_{N}` is explicit and filesystem-navigable.

**Industry comparison:** MLflow and W&B use opaque run IDs. Hydra auto-generates `outputs/YYYY-MM-DD/HH-MM-SS/` directories which are human-readable but not content-addressed. KD-GAT's approach is better for a pipeline where you need to resume runs and locate checkpoints by their config identity.

## Weaknesses

### Critical (blocks correctness or causes data loss)

#### C1. Dual merge semantics can silently diverge

**Severity: High | Probability: Low (currently) but increases with config complexity**

There are three merge sites (architecture doc, lines 140-148):

| Site | Implementation | Purpose |
|------|----------------|---------|
| `GraphIDSCLI` (jsonargparse) | Type-aware deep merge, knows about `class_path`/`init_args`, handles subclass resolution | **Instantiation** |
| `resolve_configs()` (`cli.py:42-47`) | `merge_yaml_chain()` -- naive `deep_merge` + `apply_dotted_overrides` | Reproducibility snapshot |
| `ConfigResolver._merge_yaml_chain()` (`resolve.py:60-64`) | Same naive merge | Cross-field validation |

The naive `deep_merge` in `yaml_utils.py:25-33` does recursive dict merge with atomic list replacement. jsonargparse does the same for dicts but handles additional semantics: `class_path` resolution, `init_args` type coercion, linked arguments, and default injection from `__init__` signatures.

**Where this can bite:** If a stage YAML sets `model.init_args.hidden_dims: [80, 40]` and a scale YAML sets `model.init_args.hidden_dims: [128, 64, 32]`, both merges produce the same result (list replacement). But if someone introduces a config that uses jsonargparse's `class_path` override (e.g., swapping a callback class), the naive merge will produce a raw dict while jsonargparse will instantiate the class -- and `ConfigResolver._validate_cross_fields()` could read stale/incorrect values from the naive merge.

**Competitor approach:** Hydra uses OmegaConf everywhere -- one merge implementation, one set of semantics. The tradeoff is that OmegaConf can be imported without torch. MMEngine's `Config.fromfile()` is similarly unified. The architecture doc acknowledges this tradeoff (lines 157-163: "login-node/dagster-worker safety vs merge fidelity").

**Recommendation:** This is a known risk that the project has consciously accepted for a valid reason (torch-free merge on login nodes). The mitigation is already partially in place: `validate.py` round-trips through jsonargparse and compares. Consider adding a CI step that asserts `naive_merge(chain) == jsonargparse_parse(chain)` for all recipe configs.

#### C2. No atomic config snapshot write

**Severity: Medium | Probability: Low**

`write_yaml()` in `yaml_utils.py:65-68` uses `path.write_text()` which is not atomic on NFS. If a SLURM job is killed between the `mkdir` and the `write_text` in `train_entrypoint.py:37-39`, you get an empty or partial `config_snapshot.yaml`. The `.claude/rules/critical-constraints.md` mentions NFS concerns but not for config writes.

**Competitor approach:** MLflow uses `os.fsync()` + `os.rename()` for atomic writes. The project's own CLAUDE.md mentions "NFS: `os.fsync()` before `rename()` for atomic writes" as a pattern.

**Recommendation:** Add `fsync` + temp-file-then-rename to `write_yaml()`. This is a 5-line fix.

### Near-term (will cause pain within weeks of active development)

#### N1. Recipe expansion is growing its own type system

**Severity: Medium**

`recipe_expand.py` defines `_KDSpec`, `_SweepSpec`, `_SelectionSpec`, `_RecipeEnvelope` (lines 11-53) which partially duplicate fields from `TrainingRunConfig` and `KDEntry` in `contracts.py`. For example:

- `_KDSpec.alpha` (line 17) mirrors `KDEntry.alpha` (contracts.py:16)
- `_KDSpec.teacher_scale` (line 18) mirrors `KDEntry.teacher_scale` (contracts.py:17)
- `_SweepSpec.scale` (line 28) uses `str | list[str]` while `TrainingRunConfig.scale` (contracts.py:42) is just `str`

The `_SweepSpec` has `extra="forbid"` but its `model_overrides: dict[str, Any]` (line 30) is an untyped escape hatch that bypasses all the careful validation in `TrainingRunConfig`.

**Competitor approach:** Hydra's sweep (`--multirun scale=small,large`) operates on the same config schema -- there's no separate "sweep spec" type. The sweep dimensions are derived from the config structure itself. Dagster's `RunRequest` similarly operates on the same config types.

**Recommendation:** Consider whether `_SweepSpec` can be a thin wrapper that expands into validated `TrainingRunConfig` instances earlier in the pipeline, rather than carrying its own parallel type system.

#### N2. Validation requires full Lightning import + parser instantiation

**Severity: Medium**

`validate.py:125-131` creates a throwaway `GraphIDSCLI` instance with `run=False` just to get a parser. This imports torch, Lightning, and all model code. The sys.argv manipulation (lines 123-124, 133-134) is a fragile workaround for jsonargparse's global state.

**Competitor approach:** Hydra's `--cfg job` dumps resolved config without importing the application code (it uses OmegaConf, not the actual classes). MMEngine's `Config.fromfile()` resolves without building models.

**Impact:** Validation can't run on login nodes without a GPU-capable environment, defeating one of the purposes of the torch-free config layer. The `--skip-lightning` flag (line 79) exists as a workaround.

**Recommendation:** This is an inherent limitation of jsonargparse's design (it needs the class to read `__init__` signatures). Consider a lightweight "schema cache" generated during CI that captures the parser state without requiring torch at validation time. Alternatively, accept this limitation and ensure `validate` always runs via SLURM (which it already does per project conventions).

#### N3. Checkpoint/EarlyStopping defaults are duplicated

**Severity: Low**

`CHECKPOINT_DEFAULTS` and `EARLY_STOPPING_DEFAULTS` are defined in `cli.py:27-39` and also appear in `defaults/trainer.yaml:8-17`. The architecture doc (lines 177-178) explicitly states "These must stay in sync." Manual sync requirements are a bug waiting to happen.

**Competitor approach:** Hydra's defaults list means the YAML is the single source. LightningCLI's `set_defaults()` is the canonical way, but then you don't need the YAML copy.

**Recommendation:** Either generate `defaults/trainer.yaml` from the Python constants (build step), or read the YAML values in `cli.py` and use them as `set_defaults()` arguments. The Python constants should not exist independently of the YAML.

#### N4. No config diff / change tracking between runs

**Severity: Low**

The system writes `config_snapshot.yaml` for reproducibility but doesn't diff it against previous runs. When debugging why run B differs from run A, you need to manually diff snapshots.

**Competitor approach:** W&B tracks config diffs between runs natively. Hydra's multirun logs each run's config separately with `--cfg` support. MLflow's `log_params()` enables parameter-level comparison.

**Recommendation:** Since the project already uses W&B (`WandbSaveConfigCallback` at `_lightning.py:26-33`), the config is already being logged. This is more of a tooling gap than an architectural one. The `config_snapshot.yaml` is sufficient for reproducibility; diffing is a UI concern.

### Long-term (architectural debt, won't bite until later)

#### L1. jsonargparse coupling limits config tooling

**Severity: Medium (long-term)**

The system is deeply coupled to jsonargparse's merge semantics, which are underdocumented and have edge cases around list handling, `class_path` resolution, and `Namespace` objects. jsonargparse is a single-maintainer project with ~2.5k GitHub stars. If it stops being maintained or introduces breaking changes, migration is painful.

**Competitor approach:** Hydra/OmegaConf (~8k stars, Meta-maintained) and MMEngine (~1.5k stars, OpenMMLab-maintained) have larger communities. However, jsonargparse is the only option that integrates with LightningCLI, and LightningCLI is the right choice for this project given the Lightning ecosystem usage.

**Mitigation:** The torch-free config layer (`cli.py`, `yaml_utils.py`) already provides an abstraction boundary. If jsonargparse needs replacement, only `_lightning.py` and `validate.py` change. The naive merge path is independent and could become the primary merge with a different instantiation layer.

#### L2. Recipe system has no schema versioning

**Severity: Low**

`ContractEnvelope` has versioning (`version: int`, `models.py:34`) but recipe YAML files (`ablation.yaml`, `smoke_test.yaml`) have no version field. If the recipe format changes, old recipe files will fail with unhelpful Pydantic errors rather than a clear migration message.

**Competitor approach:** Kubernetes-style `apiVersion` fields. Dagster's asset versioning. Even a comment header would help.

**Recommendation:** Add a `version: 1` field to `_RecipeEnvelope` and validate it. Low effort, prevents future confusion.

#### L3. No dry-run mode for pipeline execution

**Severity: Low**

The validation path (`Route C`) checks config parsing but doesn't simulate the full pipeline (SLURM submission, checkpoint resolution, upstream dependency resolution). You can't answer "what would dagster do with this recipe?" without actually running it.

**Competitor approach:** Dagster has `execute_in_process()` for local testing. Hydra has `--cfg job` for config-only resolution. Argo Workflows has `--dry-run`.

**Recommendation:** `enumerate_assets()` already returns the full plan. Consider a `--dry-run` flag on the orchestrator that prints the execution plan (asset names, config files, resource specs, dependency graph) without submitting.

## Comparison Matrix

| Dimension | KD-GAT | LightningCLI (vanilla) | Hydra/OmegaConf | MMEngine | Dagster Config |
|---|---|---|---|---|---|
| **Config format** | YAML (jsonargparse) | YAML (jsonargparse) | YAML (OmegaConf) | Python or YAML | Pydantic `Config` |
| **Merge semantics** | Deep dict merge, atomic list replace (two implementations) | Deep dict merge, atomic list replace (single: jsonargparse) | Deep dict merge, list append/replace via `ListMergeMode` | Dict merge with `_delete_` key for replacement | N/A (no merge -- single config per run) |
| **Type safety** | `__init__` signatures + Pydantic `extra="forbid"` + `TypedDict` | `__init__` signatures only | Structured configs (`@dataclass`) or untyped YAML | Registry `type` field + runtime validation | Pydantic `BaseModel` with full type checking |
| **CLI generation** | Auto from `__init__` + `link_arguments` | Auto from `__init__` | Auto from config groups + overrides | argparse-based `tools/train.py` | Launchpad UI + `dagster job execute` |
| **Config composition** | `--config` chain (L-R override) + `--key=val` | Same | `defaults` list + config groups + overrides | `_base_` inheritance + `_delete_` | N/A |
| **Torch-free merge** | Yes (`yaml_utils.py`) | No (requires torch for type resolution) | Yes (OmegaConf is pure Python) | No (registry needs torch) | Yes (Pydantic only) |
| **Orchestrator integration** | Dagster Component -> `TrainingSpec` -> SLURM -> `run_lightning()` | None built-in | Hydra Launcher plugins (submitit, Ray) | None built-in | Native (Definitions, assets, resources) |
| **Sweep support** | Recipe YAML -> `expand_recipe_configs()` -> combinatorial expansion | None (use external sweeper) | `--multirun` + Sweeper plugins (Optuna, Nevergrad) | None built-in | Partitions + dynamic partitions |
| **Reproducibility** | `config_snapshot.yaml` + `identity_hash` + W&B logging | `config.yaml` saved by `SaveConfigCallback` | `outputs/` dir with `.hydra/config.yaml` + `overrides.yaml` | `work_dir` with full config dump | Asset materialization metadata |
| **Validation** | Import-time assertions + `validate_recipe` (parse-only CLI) + `ConfigResolver` cross-field checks | None beyond jsonargparse type checking | `--cfg job` for resolution check | `Config.fromfile()` + registry validation | `@asset_check` + `build_defs()` validation |
| **Multi-stage DAG** | `STAGE_DEPENDENCIES` in `topology.py` + dagster asset graph | Not supported | Not supported (single-job framework) | Not supported | Native asset dependency graph |

## Recommendations

### Priority 1: Reconcile the dual merge (addresses C1)

Add a CI/test step that validates merge equivalence:

```python
def test_merge_equivalence():
    """Assert naive merge matches jsonargparse merge for all recipe configs."""
    for spec in enumerate_assets(PIPELINE_YAML, recipe):
        naive = merge_yaml_chain(spec.config_files, overrides)
        jp_parsed = parser.parse_args(["--config", snapshot])
        jp_dict = yaml.safe_load(parser.dump(jp_parsed))
        # Compare the model.init_args and data.init_args subtrees
        assert naive["model"]["init_args"] == jp_dict["model"]["init_args"]
```

This doesn't eliminate the dual merge but provides a safety net. Follow the Hydra pattern of one merge implementation only if you can solve the torch-free constraint (unlikely without abandoning jsonargparse).

### Priority 2: Atomic config writes (addresses C2)

Replace `write_yaml()` with an atomic variant:

```python
def write_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".yaml.tmp")
    with open(tmp, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)
```

This follows the project's own NFS safety conventions documented in CLAUDE.md.

### Priority 3: Eliminate callback default duplication (addresses N3)

Read the YAML values in `cli.py` instead of hardcoding them:

```python
from graphids.config.yaml_utils import read_yaml
from graphids.config.base import CONFIG_DIR

_trainer_defaults = read_yaml(CONFIG_DIR / "defaults" / "trainer.yaml")
CHECKPOINT_DEFAULTS = _trainer_defaults.get("checkpoint", {})
EARLY_STOPPING_DEFAULTS = _trainer_defaults.get("early_stopping", {})
```

This makes `defaults/trainer.yaml` the single source of truth. Follow the Hydra principle: YAML is the config, code reads it.

### Priority 4: Add recipe versioning (addresses L2)

Add `version: 1` to `_RecipeEnvelope` and validate on load:

```python
class _RecipeEnvelope(BaseModel):
    version: int = 1  # Required starting now
    # ... existing fields ...

    @field_validator("version")
    @classmethod
    def _check_version(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"Recipe version {v} not supported. Migrate to version 1.")
        return v
```

### Priority 5: Add execution plan dry-run (addresses L3)

Extend `validate_recipe()` or add a new command that prints the full execution plan:

```
python -m graphids.orchestrate plan --recipe recipes/ablation.yaml --dataset hcrl_ch
```

Output: asset names, config file chains, resource specs, dependency edges, estimated SLURM time. This already has all the ingredients in `enumerate_assets()` and `get_resources()`.

### Not recommended

- **Migrating to Hydra:** The jsonargparse/LightningCLI integration is load-bearing. Hydra would require abandoning `link_arguments`, `add_lightning_class_args`, `subclass_mode`, and the `--config` chain pattern. The migration cost exceeds the benefit, especially since the torch-free config layer already provides the main advantage Hydra would offer (pure-Python config resolution).
- **Migrating to MMEngine:** Even less appropriate -- MMEngine's registry system is designed for monolithic training scripts, not multi-stage DAG pipelines with SLURM orchestration.
- **Adding OmegaConf as a merge layer:** OmegaConf and jsonargparse have incompatible merge semantics (OmegaConf's `ListMergeMode`, interpolation syntax, etc.). Mixing them would create a third merge implementation rather than eliminating the dual merge.

## Appendix: File Index

| File | Lines | Role | Torch dependency |
|------|-------|------|-----------------|
| `cli.py` | 54 | Shared constants + torch-free entry points | None |
| `_lightning.py` | 90 | GraphIDSCLI + CLI_KWARGS | torch, Lightning |
| `__main__.py` | 110 | CLI dispatch | torch (lazy, in `_run_lightning`) |
| `config/yaml_utils.py` | 68 | YAML read/merge/write | None |
| `config/topology.py` | 114 | Stage DAG + import-time validation | None |
| `config/contracts.py` | 117 | TrainingRunConfig + KDEntry (Pydantic) | None |
| `config/recipe_expand.py` | 179 | Recipe envelope expansion | None |
| `config/runtime.py` | 41 | Env var constants | None |
| `config/paths.py` | 161 | PathContext + identity hashing | None (hashlib lazy) |
| `core/contracts/models.py` | 36 | TrainingSpec + ContractEnvelope (Pydantic) | None |
| `core/contracts/ops.py` | 147 | TrainingContract operations | None |
| `core/train_entrypoint.py` | 51 | Pipeline -> LightningCLI bridge | torch (at call time) |
| `orchestrate/resolve.py` | 230 | ConfigResolver (cross-field validation) | None |
| `orchestrate/validate.py` | 198 | Full config chain validation | torch (GraphIDSCLI import) |
| `orchestrate/planning.py` | 172 | Recipe -> StageConfig expansion | None |
