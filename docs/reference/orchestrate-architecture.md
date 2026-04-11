I've got the full picture now. Here's my analysis + proposal.

## Current state map

| File                               | Lines | What it owns                                                                                                                                                                                   | Consumers                                                                         |
| ---------------------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `graphids/instantiate.py`          | 220   | `Instantiator` class with 9 static/class methods + `InstantiatedRun` dataclass + `instantiate` alias + dead `_init_kwargs`                                                                     | `stage.build`, `cli/_slurm.py::probe_budget` (via `build_model_from_spec`), tests |
| `orchestrate/__init__.py`          | 12    | Docstring only (post-cleanup)                                                                                                                                                                  | —                                                                                 |
| `orchestrate/_setup.py`            | 50    | `ensure_spawn()` + `touch_marker()` (unrelated helpers)                                                                                                                                        | `run.py`, `stage.py`, `cli/_training.py`                                          |
| `orchestrate/planning/__init__.py` | 20    | Re-exports from planner.py + recipes.py                                                                                                                                                        | `run.py`, `cli/_slurm.py`, tests                                                  |
| `orchestrate/planning/planner.py`  | 176   | `StageConfig` model + `StageConfig.from_recipe` + `enumerate_assets` + `resolve_jsonnet_path`                                                                                                  | `run.py::build_pipeline_stages`, `actors.py` (deleted), tests                     |
| `orchestrate/planning/recipes.py`  | 133   | `KDEntry` + `TrainingRunConfig` + `expand_recipe_configs` one-liner                                                                                                                            | `planner.py`, `run.py`                                                            |
| `orchestrate/resolve.py`           | 134   | `_build_tla_dict` (private) + `ResolvedConfig` dataclass + `.resolve()` classmethod with inline monitor-mode warning                                                                           | `run.py::_run_one_stage`, tests                                                   |
| `orchestrate/stage.py`             | 128   | `build(rendered, validated)` + `train(artifacts, ...)` + `evaluate(artifacts, ...)` — the dumb primitives                                                                                      | `run.py::_run_one_stage`, `cli/_training.py::fit/test`                            |
| `orchestrate/analyze.py`           | 45    | `run_single_analysis(spec)` — thin wrapper over `core/analysis/Analyzer`                                                                                                                       | `run.py::_run_one_stage`                                                          |
| `orchestrate/run.py`               | 262   | `PipelineConfig` (CLI schema) + `build_pipeline_stages` (planner bridge) + `PipelineResult` + `_run_one_stage` (the big verb) + `run_pipeline` (the loop) + `_ANALYZABLE_MODEL_TYPES` constant | `cli/_pipeline.py`                                                                |

**Total: 1,180 lines across 10 Python files.** Call graph:

```
cli/_pipeline.py::pipeline_run
  └─> run.py::run_pipeline
        ├─> _setup.py::ensure_spawn
        ├─> planning/*::build_pipeline_stages → StageConfig list
        │     └─> planning/recipes.py::TrainingRunConfig, KDEntry
        │     └─> planning/planner.py::enumerate_assets
        └─> [per stage, with retry]
            run.py::_run_one_stage
              ├─> resolve.py::ResolvedConfig.resolve
              │     ├─> resolve.py::_build_tla_dict  ← private, lives in wrong place
              │     ├─> config/jsonnet.py::render
              │     └─> config/schemas.py::validate_config
              ├─> [skip if .complete marker]
              │     └─> [return; no primitives called]
              ├─> stage.py::build
              │     └─> graphids/instantiate.py::Instantiator.build_run
              │           ├─> build_model_from_config ─┐
              │           ├─> build_datamodule          │ all on Instantiator class
              │           ├─> build_trainer             │ all @classmethod
              │           ├─> build_callbacks           │ no state
              │           └─> build_loggers           ─┘
              ├─> stage.py::train
              │     ├─> _otel.py::wire_file_exporters
              │     ├─> trainer.fit()
              │     └─> _setup.py::touch_marker(.train_complete)
              ├─> stage.py::evaluate
              │     ├─> _otel.py::wire_file_exporters  ← DUPLICATE call, already wired in train
              │     ├─> trainer.test()
              │     ├─> _setup.py::touch_marker(.test_complete)
              │     └─> _setup.py::touch_marker(.complete)
              └─> analyze.py::run_single_analysis
                    └─> core/analysis/Analyzer(**spec).run()
```

## Problems — concrete, not vague

**P1.** ~~`instantiate.py` is at the top level but only `stage.build` + one CLI consume it.~~ **RESOLVED** — moved to `graphids/orchestrate/instantiate.py` during the orchestrate consolidation pass. No longer a peer of `graphids.core` / `graphids.config`.

**P2. `Instantiator` is a class that holds no state.** Nine `@classmethod` / `@staticmethod` methods with `cls.` prefixes everywhere. It's a namespace spelled as a class. The module-level `instantiate = Instantiator.build_run` alias admits this — the class exists only to group the methods.

**P3. `build_model` and `build_block` are two different instantiation paths for similar work.** `build_block` recursively resolves nested `{class_path, init_args}` blocks; `build_model` does kwarg filtering but no recursion. The split exists because `build_model_from_config` calls `inject_loss_fn` for KD, which `build_block` doesn't know about. Net effect: callers can't use one path uniformly.

**P4. Dead code.** `_init_kwargs` at `instantiate.py:208` duplicates the accepted-param-extraction logic from `filter_kwargs`. Nothing in the repo imports it.

**P5.** ~~`_build_tla_dict` lives in `resolve.py` but is a projection of `StageConfig` fields.~~ **RESOLVED** — the logic is now `StageConfig.to_tla_dict(dataset, seed, run_dir, upstream_ckpts, ckpt_path)` in `orchestrate/config.py`. Adding a new TLA means editing `to_tla_dict` + the jsonnet signature — two places, co-located with the schema.

**P6. `ResolvedConfig.resolve()` does cross-field validation inline.** Lines 121-132 emit a log warning if `validated.checkpoint_monitor/mode` doesn't match the stage family convention. The plan (per `.claude/rules/config-system.md`) says this should move into `validate_config`. It hasn't. Drift.

**P7. `run.py` is three concerns bolted together.** (a) `PipelineConfig` Pydantic schema + `build_pipeline_stages` planner bridge (~130 lines), (b) `_run_one_stage` per-stage verb + `_ANALYZABLE_MODEL_TYPES` constant + analyze wiring (~80 lines), (c) `run_pipeline` driver loop (~50 lines). Three different change reasons (CLI schema evolution, per-stage semantics, retry/DAG policy) in one file.

**P8. `orchestrate/analyze.py` is a 45-line shim whose body just delegates to `graphids.core.analysis.Analyzer`.** The analyzer lives elsewhere; this file is the awkward middle layer between "orchestrate runs analyzers" and "core/analysis defines them." It should be on the `core/analysis` side.

**P9. `stage.py::build/train/evaluate` don't know about `ResolvedConfig`, so `_run_one_stage` has to unpack it.** The rationale was "stage primitives should be callable from the CLI `fit`/`test` commands which don't have a full ResolvedConfig" — but looking at `cli/_training.py:29-38`, the CLI path _does_ call `render_config` + `validate_config` + `build` with `(rendered, validated)` pairs. The CLI already has everything a `ResolvedConfig` holds. The distinction is artificial.

**P10. `_run_one_stage` calls `wire_file_exporters(run_dir)` twice** — once inside `stage.train`, once inside `stage.evaluate`. Idempotent but wasteful. Should wire once per stage in `_run_one_stage` itself.

**P11. `planning/planner.py` + `planning/recipes.py` are one logical module split across two files totaling 309 lines.** They cross-reference each other (`planner.py` imports `TrainingRunConfig` from recipes.py). The `planning/` sub-package exists to hold two files that could be one.

**P12. `_setup.py` bundles two unrelated helpers.** `ensure_spawn` is process-lifecycle; `touch_marker` is NFS-safe file I/O. Different change reasons, different consumers.

**P13. `expand_recipe_configs` (`recipes.py:119-133`) is a one-line wrapper around `render()` with a hardcoded TLA dict.** It only exists to give the operation a name. It has zero callers in the current orchestrate path — nothing in the main pipeline invokes it.

**P14. `_ANALYZABLE_MODEL_TYPES = frozenset({"vgae", "dgi", "gat"})` lives in `run.py` but the knowledge belongs to the analyzer module.** If you add a new analyzable model (say, DGI variants), you'd edit `core/analysis/Analyzer` AND this constant in `run.py`. Two places, will drift.

## Proposed design — 6 clean layers

```
Layer 6: CLI BINDING         cli/_pipeline.py, cli/_training.py
                             (Typer commands — thin wrappers, no logic)
                                      ↑
Layer 5: PIPELINE DRIVER     orchestrate/run.py
                             run_pipeline, _run_one_stage
                             (loop + retry + analyze + marker)
                                      ↑
Layer 4: STAGE PRIMITIVES    orchestrate/stage.py
                             build, train, evaluate
                             (one verb per stage; takes ResolvedConfig)
                                      ↑
Layer 3: INSTANTIATION       orchestrate/instantiate.py
                             build_run, build_model, build_datamodule,
                             build_trainer, build_callbacks, build_loggers
                             (rendered dict → (trainer, model, datamodule))
                                      ↑
Layer 2: CONFIG RESOLUTION   orchestrate/resolve.py
                             resolve_config(cfg, ...) → ResolvedConfig
                             (StageConfig + runtime context → rendered + validated)
                                      ↑
Layer 1: PLANNING            orchestrate/planning.py
                             enumerate_assets, build_pipeline_stages
                             (recipe / PipelineConfig → list[StageConfig])
                                      ↑
Layer 0: DATA TYPES          orchestrate/config.py
                             PipelineConfig, StageConfig, TrainingRunConfig,
                             KDEntry, ResolvedConfig, InstantiatedRun, PipelineResult
                             (Pydantic / frozen dataclass, no side effects)
```

Each layer:

- Depends only on layers below it
- Has one public entry point
- Has one file (except Layer 6, which is two commands in two files)
- Does not cross-import with its peers

### The single invariant

**`StageConfig` is the central data type.** It's the boundary between planning and execution. Above it (Layer 1), planners produce lists of `StageConfig`. Below it (Layer 2), resolvers turn one `StageConfig` + runtime context into a `ResolvedConfig`. Every TLA, every jsonnet binding, every cross-stage identity decision flows through `StageConfig`.

The refactor makes this invariant visible by putting ALL the TLA-building logic on `StageConfig` itself.

## File-by-file plan

### DELETE

| File                                                 | Why                                                             | Where content goes                                                                                       |
| ---------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `graphids/instantiate.py`                            | Top-level location is wrong; only orchestrate + one CLI use it  | → `orchestrate/instantiate.py`                                                                           |
| `orchestrate/planning/__init__.py`                   | Re-export shim for a 2-file sub-package that should be 1 file   | → `orchestrate/planning.py` package-less                                                                 |
| `orchestrate/planning/planner.py`                    | Merged into flat `planning.py`                                  | → `orchestrate/planning.py`                                                                              |
| `orchestrate/planning/recipes.py`                    | Merged into flat `planning.py`                                  | → `orchestrate/planning.py`                                                                              |
| `orchestrate/analyze.py`                             | Shim over `core/analysis.Analyzer` belongs next to the analyzer | → `core/analysis/runner.py`                                                                              |
| `orchestrate/_setup.py`                              | Bundles unrelated helpers                                       | `ensure_spawn` → `orchestrate/_spawn.py` (or absorbed into `run.py`); `touch_marker` → `graphids/_fs.py` |
| `Instantiator` class wrapper inside `instantiate.py` | Namespace-spelled-as-class with no state                        | Each method becomes a module-level function                                                              |
| `_init_kwargs` function at `instantiate.py:208`      | Dead code (duplicate of `filter_kwargs` logic)                  | Gone                                                                                                     |
| `_build_tla_dict` function at `resolve.py:22`        | Projection of StageConfig fields                                | Becomes `StageConfig.to_tla_dict()` method                                                               |
| `expand_recipe_configs` at `recipes.py:119`          | Unused one-liner wrapper around `render()`                      | Gone; callers inline the `render()` call                                                                 |
| `_ANALYZABLE_MODEL_TYPES` at `run.py:41`             | Knowledge belongs to analyzer                                   | Moves to `core/analysis/runner.py` as `ANALYZABLE_MODEL_TYPES` export                                    |
| Monitor/mode warning at `resolve.py:121-132`         | Should live in `validate_config` per plan                       | → `config/schemas.py::_monitor_pair_matches_stage_family` validator                                      |

### ADD

| File                                            | Lines (est) | Contents                                                                                                                                                                                                                                                            |
| ----------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `orchestrate/config.py`                         | ~200        | All frozen data types in one file: `PipelineConfig`, `StageConfig` (with `.to_tla_dict()`), `TrainingRunConfig`, `KDEntry`, `ResolvedConfig`, `InstantiatedRun`, `PipelineResult`. Re-export `PathContext` from `config/topology`.                                  |
| `orchestrate/planning.py`                       | ~200        | Flat file: `enumerate_assets`, `build_pipeline_stages`, `resolve_jsonnet_path`. Imports types from `orchestrate/config.py`.                                                                                                                                         |
| `orchestrate/instantiate.py`                    | ~180        | Flat module, no class wrapper: `build_run`, `build_model`, `build_datamodule`, `build_trainer`, `build_callbacks`, `build_loggers`, `build_block`, `import_class`, `filter_kwargs`. Single `build_model_from_config` path (no `build_model` / `build_block` split). |
| `orchestrate/resolve.py`                        | ~55         | Only `resolve_config(cfg, *, lake_root, user, dataset, seed, upstream_ckpts)` → `ResolvedConfig`. No private TLA packer; that's on `StageConfig` now. No cross-field check; that's in `validate_config` now.                                                        |
| `orchestrate/stage.py`                          | ~90         | `build(resolved)`, `train(artifacts, resolved, *, resume_from=None)`, `evaluate(artifacts, resolved)`. Each takes `ResolvedConfig` directly. `wire_file_exporters` called once by caller, not per primitive.                                                        |
| `orchestrate/run.py`                            | ~130        | Only `run_pipeline(config)` + `_run_one_stage(cfg, ...)`. Imports `PipelineConfig`/`PipelineResult` from `config.py`, imports `build_pipeline_stages` from `planning.py`, imports `run_single_analysis` from `core/analysis/runner`.                                |
| `core/analysis/runner.py`                       | ~55         | `run_single_analysis(spec)` + `ANALYZABLE_MODEL_TYPES` frozenset. Moved from `orchestrate/analyze.py`.                                                                                                                                                              |
| `graphids/_fs.py`                               | ~20         | `touch_marker(path)` only. NFS-safe fsync of file + parent dir.                                                                                                                                                                                                     |
| `orchestrate/_spawn.py` OR absorb into `run.py` | ~15         | `ensure_spawn()`. Used once per pipeline invocation. Could even be inlined into `run.py::run_pipeline`.                                                                                                                                                             |
| `orchestrate/__init__.py`                       | ~30         | Re-exports the public API: `PipelineConfig`, `PipelineResult`, `StageConfig`, `run_pipeline`, `resolve_config`, `build_run`.                                                                                                                                        |

### MOVE

- `cli/_slurm.py::probe_budget` currently calls `Instantiator.build_model_from_spec`. Either:
  - **Option A**: Keep the function in `orchestrate/instantiate.py` as `build_model_from_spec(model_type, scale, ...)`. Update CLI import. Simple rename.
  - **Option B**: Move it to `graphids/core/models/factory.py` (new file) since "instantiate a model from (type, scale)" is a model concern. Cleaner layering — probe-budget doesn't need to reach into orchestrate.
  - **Recommendation**: Option B. It's 25 lines that unambiguously belong to core/models.

## Data flow diagram (after refactor)

```
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 0 — DATA TYPES (orchestrate/config.py)                      │
│                                                                   │
│   PipelineConfig ──┐                                              │
│                    │                                              │
│   TrainingRunConfig ── KDEntry                                    │
│           │                                                       │
│           ↓                                                       │
│   StageConfig      ← central boundary type                        │
│     .to_tla_dict(dataset, seed, run_dir, upstream_ckpts)          │
│                                                                   │
│   ResolvedConfig(paths: PathContext, validated, rendered)         │
│                                                                   │
│   InstantiatedRun(trainer, model, datamodule, merged)             │
│                                                                   │
│   PipelineResult(checkpoints, analyzed_assets, stage_to_asset)    │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ imports types from Layer 0
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 1 — PLANNING (orchestrate/planning.py)                      │
│                                                                   │
│   enumerate_assets(recipe: dict) → list[StageConfig]              │
│   build_pipeline_stages(cfg: PipelineConfig) → list[StageConfig]  │
│   resolve_jsonnet_path(stage: str) → str                          │
│                                                                   │
│   ← no side effects, no torch, no jsonnet subprocess              │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 2 — CONFIG RESOLUTION (orchestrate/resolve.py)              │
│                                                                   │
│   resolve_config(                                                 │
│       cfg: StageConfig, *,                                        │
│       lake_root, user, dataset, seed, upstream_ckpts,             │
│   ) → ResolvedConfig                                              │
│                                                                   │
│   ← shells out to jsonnet binary, runs validate_config            │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 3 — INSTANTIATION (orchestrate/instantiate.py)              │
│                                                                   │
│   build_run(rendered, validated) → InstantiatedRun                │
│   (flat module — no Instantiator class wrapper)                   │
│                                                                   │
│   ← imports torch, does importlib class imports                   │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 4 — STAGE PRIMITIVES (orchestrate/stage.py)                 │
│                                                                   │
│   build(resolved: ResolvedConfig) → InstantiatedRun               │
│   train(artifacts, resolved, *, resume_from=None) → Path          │
│   evaluate(artifacts, resolved) → dict                            │
│                                                                   │
│   ← GPU reset, OTel wiring, trainer.fit/test, marker touch        │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 5 — PIPELINE DRIVER (orchestrate/run.py)                    │
│                                                                   │
│   run_pipeline(config: PipelineConfig) → PipelineResult           │
│   _run_one_stage(cfg: StageConfig, ...) → (ckpt_path, analyzed)   │
│                                                                   │
│   ← loops, retries, handles skip check, calls run_single_analysis │
└───────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │
┌───────────────────────────────────────────────────────────────────┐
│ LAYER 6 — CLI BINDING (cli/_pipeline.py, cli/_training.py)        │
│                                                                   │
│   @app.command("pipeline-run") pipeline_run(...) → run_pipeline   │
│   @app.command("fit") fit(...) → resolve → build → train          │
│   @app.command("test") test(...) → resolve → build → evaluate     │
│                                                                   │
│   ← thin Typer wrappers, no logic                                 │
└───────────────────────────────────────────────────────────────────┘
```

## Key signature changes (with examples)

### `StageConfig.to_tla_dict()` as the single TLA projection

```python
# orchestrate/config.py

class StageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    model_type: str
    scale: str
    model_init_overrides: dict[str, Any] = Field(default_factory=dict)
    identity: str = ""
    resource_model: str = ""
    kd_overrides: dict[str, Any] = Field(default_factory=dict)
    trainer_overrides: dict[str, Any] = Field(default_factory=dict)
    stage_overrides: dict[str, Any] = Field(default_factory=dict)
    resource_overrides: dict[str, str | int] = Field(default_factory=dict)
    upstream_asset_names: tuple[str, ...] = ()
    upstream_model_families: dict[str, str] = Field(default_factory=dict)

    @computed_field
    @property
    def asset_name(self) -> str: ...

    @computed_field
    @property
    def jsonnet_path(self) -> str: ...

    def to_tla_dict(
        self,
        *,
        dataset: str,
        seed: int,
        run_dir: str,
        upstream_ckpts: dict[str, str],
        ckpt_path: str | None = None,
    ) -> dict[str, Any]:
        """Pack this StageConfig + runtime context into the jsonnet TLA dict.

        This is the ONLY place field names map to jsonnet TLA keys.
        Adding a new TLA means editing this method + the stage jsonnet signature.
        """
        from graphids.config.topology import TOPOLOGY

        tla: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "run_dir": run_dir,
            "scale": self.scale,
            "trainer_overrides": dict(self.trainer_overrides),
            "stage_overrides": dict(self.stage_overrides),
        }
        tla.update(self.model_init_overrides)

        stage_def = TOPOLOGY.stages.get(self.stage)
        accepted = set(stage_def.stage_tlas) if stage_def else set()

        if "fusion_method" in accepted:
            tla["fusion_method"] = self.resource_model or self.model_type

        for upstream_asset, ckpt in upstream_ckpts.items():
            family = self.upstream_model_families.get(upstream_asset)
            if family == "unsupervised" and "vgae_ckpt_path" in accepted:
                tla["vgae_ckpt_path"] = ckpt
            elif family == "supervised" and "gat_ckpt_path" in accepted:
                tla["gat_ckpt_path"] = ckpt

        if "distillation_config" in accepted:
            tla["distillation_config"] = dict(self.kd_overrides) if self.kd_overrides else None

        if ckpt_path is not None and "ckpt_path" in accepted:
            tla["ckpt_path"] = ckpt_path

        return tla
```

### `resolve_config()` as a free function, not a classmethod

```python
# orchestrate/resolve.py — ~55 lines total

from __future__ import annotations
from graphids.config.jsonnet import render
from graphids.config.schemas import validate_config
from graphids.config.topology import PathContext
from graphids.orchestrate.config import ResolvedConfig, StageConfig


def resolve_config(
    cfg: StageConfig,
    *,
    lake_root: str,
    user: str,
    dataset: str,
    seed: int,
    upstream_ckpts: dict[str, str] | None = None,
) -> ResolvedConfig:
    """Render + validate a StageConfig into a ResolvedConfig.

    Single entry point for the config resolution layer. Shells out to
    the jsonnet binary via render(), then runs validate_config() to
    gate on Pydantic cross-field checks (including the stage/monitor
    family convention, now enforced inside validate_config).
    """
    paths = PathContext(
        lake_root=lake_root, user=user, dataset=dataset,
        model_type=cfg.model_type, scale=cfg.scale, stage=cfg.stage,
        identity=cfg.identity, kd_tag=cfg.kd_tag, seed=seed,
    )
    tla = cfg.to_tla_dict(
        dataset=dataset, seed=seed, run_dir=str(paths.run_dir),
        upstream_ckpts=upstream_ckpts or {},
    )
    rendered = render(cfg.jsonnet_path, tla)
    validated = validate_config(rendered)  # cross-field check now lives inside
    return ResolvedConfig(paths=paths, validated=validated, rendered=rendered)
```

Changes vs current `resolve.py`:

- 55 lines instead of 134
- No private `_build_tla_dict` — logic moved onto `StageConfig.to_tla_dict`
- No inline monitor/mode warning — moved into `validate_config`
- No classmethod ceremony — just a free function

### `stage.py` primitives take `ResolvedConfig` directly

```python
# orchestrate/stage.py — ~90 lines total

from __future__ import annotations
from pathlib import Path
import gc
import torch

from graphids._otel import get_logger
from graphids._fs import touch_marker
from graphids.config.constants import PHASE_MARKERS, COMPLETE_MARKER
from graphids.orchestrate.config import InstantiatedRun, ResolvedConfig
from graphids.orchestrate.instantiate import build_run

log = get_logger(__name__)


def build(resolved: ResolvedConfig) -> InstantiatedRun:
    """Instantiate trainer + model + datamodule from a resolved config.

    GPU state is reset first so a prior stage's VRAM / compiled kernels
    don't leak into this one.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    torch.compiler.reset()
    return build_run(resolved.rendered, validated=resolved.validated)


def train(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
    *,
    resume_from: str | None = None,
) -> Path:
    """Fit the model and return the canonical checkpoint path.

    Touches the train phase marker on success. Caller is expected to
    have wired OTel file exporters for this stage's run_dir already.
    """
    stage_name = resolved.paths.stage
    ckpt_file = Path(str(resolved.paths.ckpt_file))
    log.info("stage_train", stage=stage_name, run_dir=str(resolved.paths.run_dir))
    artifacts.trainer.fit(
        artifacts.model,
        datamodule=artifacts.datamodule,
        ckpt_path=resume_from,
    )
    touch_marker(Path(str(resolved.paths.run_dir)) / PHASE_MARKERS["train"])
    log.info("stage_train_complete", stage=stage_name, ckpt=str(ckpt_file))
    return ckpt_file


def evaluate(
    artifacts: InstantiatedRun,
    resolved: ResolvedConfig,
) -> dict:
    """Run the test phase and return metrics. Lenient on failure.

    Touches the test phase marker on success and the complete marker
    unconditionally. Caller is expected to have wired OTel file
    exporters already.
    """
    stage_name = resolved.paths.stage
    run_dir = Path(str(resolved.paths.run_dir))
    ckpt_file = Path(str(resolved.paths.ckpt_file))
    try:
        log.info("stage_test", stage=stage_name)
        metrics = artifacts.trainer.test(
            artifacts.model, datamodule=artifacts.datamodule,
            ckpt_path=str(ckpt_file),
        )
        touch_marker(run_dir / PHASE_MARKERS["test"])
        result = metrics or {}
    except Exception as exc:
        log.warning("stage_test_failed", stage=stage_name, error=str(exc))
        result = {}
    touch_marker(run_dir / COMPLETE_MARKER)
    log.info("stage_complete", stage=stage_name)
    return result
```

Changes:

- `train` and `evaluate` each take 2 args (`artifacts`, `resolved`) instead of 4–5. Run dir, ckpt file, stage name all come from `resolved.paths`.
- `wire_file_exporters` no longer called here — the caller (`_run_one_stage`) wires it once per stage.
- `build` takes `ResolvedConfig` instead of two separate args (`rendered`, `validated`).

### `_run_one_stage` becomes much shorter

```python
# orchestrate/run.py — _run_one_stage shrinks

def _run_one_stage(
    cfg: StageConfig,
    *,
    dataset: str, seed: int, lake_root: str, user: str,
    upstream_ckpts: dict[str, str],
) -> tuple[str, bool]:
    from graphids._otel import wire_file_exporters
    from graphids.core.analysis.runner import (
        ANALYZABLE_MODEL_TYPES, run_single_analysis,
    )
    from graphids.core.analysis.schemas import AnalysisSpec
    from graphids.config.constants import PHASE_MARKERS
    from graphids._fs import touch_marker
    from graphids.orchestrate.resolve import resolve_config
    from graphids.orchestrate.stage import build, train, evaluate

    resolved = resolve_config(
        cfg, lake_root=lake_root, user=user,
        dataset=dataset, seed=seed, upstream_ckpts=upstream_ckpts,
    )
    run_dir = Path(str(resolved.paths.run_dir))
    ckpt_file = Path(str(resolved.paths.ckpt_file))

    if ckpt_file.exists() and resolved.paths.complete_marker.exists():
        log.info("stage_skip_complete", stage=cfg.stage, run_dir=str(run_dir))
        return str(ckpt_file), (run_dir / PHASE_MARKERS["analyze"]).exists()

    wire_file_exporters(run_dir)                 # once, not twice
    artifacts = build(resolved)                   # resolved, not (rendered, validated)
    train(artifacts, resolved)
    evaluate(artifacts, resolved)

    analyzed = False
    if cfg.model_type in ANALYZABLE_MODEL_TYPES:  # imported from core/analysis/runner
        try:
            run_single_analysis(AnalysisSpec(
                ckpt_path=str(ckpt_file), dataset=dataset,
                model_type=cfg.model_type,
                output_dir=str(ckpt_file.resolve().parent.parent / "artifacts"),
                seed=seed,
            ))
            touch_marker(run_dir / PHASE_MARKERS["analyze"])
            analyzed = True
        except Exception as exc:
            log.warning("stage_analyze_failed", stage=cfg.stage, error=str(exc))

    return str(ckpt_file), analyzed
```

Changes:

- 4 positional args to `stage.*` primitives collapse into 1 (`resolved`)
- `wire_file_exporters` called once instead of twice (inside train + inside evaluate)
- `_ANALYZABLE_MODEL_TYPES` imported from core/analysis, not defined in run.py
- Function shrinks from ~80 lines to ~45 lines

## Before/after line count

| Area                                 | Before    | After           | Δ        |
| ------------------------------------ | --------- | --------------- | -------- |
| `graphids/instantiate.py`            | 220       | 0 (moved)       | −220     |
| `orchestrate/instantiate.py`         | 0         | 180             | +180     |
| `orchestrate/__init__.py`            | 12        | 30 (re-exports) | +18      |
| `orchestrate/_setup.py`              | 50        | 0 (split)       | −50      |
| `orchestrate/_spawn.py` (or inlined) | 0         | 15              | +15      |
| `graphids/_fs.py`                    | 0         | 20              | +20      |
| `orchestrate/planning/__init__.py`   | 20        | 0 (deleted)     | −20      |
| `orchestrate/planning/planner.py`    | 176       | 0 (merged)      | −176     |
| `orchestrate/planning/recipes.py`    | 133       | 0 (merged)      | −133     |
| `orchestrate/planning.py`            | 0         | 200             | +200     |
| `orchestrate/config.py`              | 0         | 200             | +200     |
| `orchestrate/resolve.py`             | 134       | 55              | −79      |
| `orchestrate/stage.py`               | 128       | 90              | −38      |
| `orchestrate/analyze.py`             | 45        | 0 (moved)       | −45      |
| `core/analysis/runner.py`            | 0         | 55              | +55      |
| `orchestrate/run.py`                 | 262       | 130             | −132     |
| **Total**                            | **1,180** | **975**         | **−205** |

**~205 lines removed, 10 files → 10 files** (same count, different shape). The gain isn't primarily line-count — it's:

1. **One invariant, one place**: `StageConfig.to_tla_dict` is the single point where field names map to jsonnet TLAs. Adding a TLA is a two-edit change instead of a three-edit change.
2. **Fewer args at every call site**: `stage.build/train/evaluate` take `ResolvedConfig` instead of unpacked tuples.
3. **Layered top-to-bottom**: you can read `orchestrate/config.py` → `planning.py` → `resolve.py` → `instantiate.py` → `stage.py` → `run.py` in order and each layer only references layers above it in the import graph.
4. **No namespace-spelled-as-class**: `Instantiator.build_run` becomes `build_run`. One import, less ceremony.
5. **No shim middleware**: `orchestrate/analyze.py` → `core/analysis/runner.py`. The analyzer now owns the "how do I run one analysis" verb.
6. **Cross-field validation where it belongs**: monitor/mode check lives in `validate_config` where the plan said it should.

## Migration order (safe, test-at-each-step)

1. **Create `orchestrate/config.py`** with all data types (copy-paste from existing files). Add `StageConfig.to_tla_dict()` method. Existing files re-export from `config.py` for backward compat temporarily. **Test**: `pytest --collect-only` + import smoke.
2. **Flatten `planning/` → `planning.py`**. Update `from graphids.orchestrate.planning import ...` callers — the public API stays the same, so most imports don't change. **Test**: `pipeline-run --dry-run --dataset hcrl_sa`.
3. **Simplify `resolve.py`**. Delete `_build_tla_dict`, use `cfg.to_tla_dict(...)`. Move monitor/mode warning into `validate_config`. Convert `ResolvedConfig.resolve` classmethod → `resolve_config` free function (keep classmethod as deprecated shim for one cycle if needed). **Test**: dry-run + collect-only.
4. **Move `graphids/instantiate.py` → `orchestrate/instantiate.py`**. Flatten to module-level functions (delete `Instantiator` class wrapper). Delete dead `_init_kwargs`. Update `stage.py` + `cli/_slurm.py` imports. **Test**: dry-run + module import check.
5. **Update `stage.py`**. Change signatures to take `ResolvedConfig`. Remove `wire_file_exporters` calls (move to caller). **Test**: dry-run + a targeted unit test for `build`.
6. **Split `run.py`**. Move `PipelineConfig` + `build_pipeline_stages` out to `config.py` + `planning.py`. Leave `run_pipeline` + `_run_one_stage`. Update `_run_one_stage` to call `wire_file_exporters` once. **Test**: dry-run + full CLI smoke.
7. **Move `orchestrate/analyze.py` → `core/analysis/runner.py`**. Export `ANALYZABLE_MODEL_TYPES` from there. Update `run.py` imports. **Test**: dry-run + `analyze` CLI command still works.
8. **Split `_setup.py`**. Move `touch_marker` to `graphids/_fs.py`. Move `ensure_spawn` to `orchestrate/_spawn.py` or inline into `run_pipeline`. **Test**: full import chain.
9. **Update `orchestrate/__init__.py`** to re-export the public API: `PipelineConfig`, `PipelineResult`, `StageConfig`, `run_pipeline`, `resolve_config`, `build_run`.
10. **Run the gpudebug smoke** — if this passes, the refactor is done and everything else is bikeshedding.

Each step is independently revertable (< 200 lines of churn). No step touches more than 3 files.

## Open questions before committing

1. **Should `build_model_from_spec` (the probe-budget path) move to `core/models/factory.py`, or stay in `instantiate.py`?** Recommendation: move. It's a model concern, not an orchestration concern, and decouples probe-budget from the orchestrate stack.

2. **Should `InstantiatedRun` carry `resolved: ResolvedConfig` instead of `merged: dict`?** The current `merged` field on `InstantiatedRun` is only used by tests. If nothing production-side needs it, drop it — or replace with `resolved` for symmetry. Recommendation: drop it (it's a debugging leftover).

3. **Should `cli/_training.py::fit` / `test` go through `resolve_config` directly?** Currently they do `render_config → validate_config → build` without producing a `ResolvedConfig`. If stage primitives take `ResolvedConfig`, the CLI path needs to construct one. This means `fit` inherits path context (run_dir derived from jsonnet's `trainer.default_root_dir`). Trivial. Recommendation: yes, make the CLI use `resolve_config` too — same code path everywhere.

4. **Does `ensure_spawn` need its own file?** It's called at `run_pipeline`'s entry and at `cli/_training.py::fit`'s entry. Two call sites. Could live in `graphids/_spawn.py` or be inlined both places. Recommendation: `graphids/_spawn.py` (15 lines) since it's cross-package.

5. **`core/analysis/runner.py` vs `core/analysis/orchestrate.py` name?** `runner.py` is clearer — the file answers "how do I run one analysis against one checkpoint." Recommendation: `runner.py`.

6. **Should `PipelineResult` live in `config.py` (data types) or `run.py` (producer)?** It's a result type, produced by `run_pipeline` and only consumed at CLI boundaries. Could go either way. Recommendation: `config.py` — then `run.py` imports everything from `config.py` and the layering stays strict.

## What this does NOT change

- The jsonnet config system and `configs/stages/*.jsonnet` files — untouched.
- `config/schemas.py::validate_config` — adds one cross-field validator, otherwise unchanged.
- `config/topology.py::PathContext` — unchanged.
- `core/trainer.py::Trainer` — unchanged.
- `core/monitoring.py::OTelTrainingCallback` — unchanged.
- The CLI command surface (`fit`, `test`, `pipeline-run`, `analyze`, etc.) — imports change, commands don't.
- The way stages share `build/train/evaluate` between CLI and pipeline driver — still shared, just taking `ResolvedConfig` instead of unpacked tuples.

---

Want me to:
(a) Write this design up as a new reference doc (`docs/reference/orchestrate-architecture.md`) so it's durable across sessions,
(b) Start the refactor from step 1 (create `config.py`), or
(c) Push back on any of the open questions first — I'm particularly interested in your answer on Q3 (CLI `fit`/`test` going through `resolve_config`)?
