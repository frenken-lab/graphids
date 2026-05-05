# Module Responsibilities

**Plan modules** (`graphids/plan/plans/`) — Python files exposing
`build(*, dataset: str, seed: int) -> list[dict]`. Each plan composes
class-path specs + composers into rows that match the `Plan`
schema.

**Composer** (`graphids/plan/compose.py`) — single `compose(...)`
plus a thin `fusion(...)` wrapper. Takes bare `{class_path, init_args}`
blocks (model, data, loss) and emits a frozen
:class:`graphids.plan.row.RowSpec` whose `rendered` is a typed
:class:`graphids.plan.blueprint.RenderedConfig` (Pydantic, frozen,
`extra="forbid"` — typo'd field at compose time raises
`ValidationError`).

**Lib** (`graphids/plan/lib.py`) — class-path string constants
(`GAT`, `VGAE`, `FOCAL`, …) + `spec(cls_path, **init_args)` helper +
the four primitives that compose / validate (`can_bus` registry
check, `graph_dm` conditional knobs, `fusion_dm` path derivation,
`curriculum` deepcopy + reduction injection). Defaults for trivial
primitives live with the model class itself (e.g. `GAT.__init__`'s
`_SCALES` table), not duplicated here.

**Pydantic / `Plan`** (`graphids/plan/blueprint.py`) —
validation gate. Each row is a discriminated union (`TrainRow` |
`CmdRow` | `ExtractRow` | `AnalyzeRow`) with `extra="forbid"`.
`TrainRow.rendered_config` is itself a typed `RenderedConfig`
(`model: ClassPath`, `data: ClassPath`, `trainer: TrainerCfg`,
`callbacks: dict[str, ClassPath]`) — validation is structural
end-to-end. Render bugs surface here before SLURM sees them.

**Orchestrate** (`graphids/orchestrate.py`) — `run_row(row)` dispatches
on `row.action` (fit/test/extract/analyze). For training rows,
`_instantiate` walks nested `class_path` blocks via importlib with
signature-filtered kwargs and returns the trainer / model / datamodule.
Owns module-level runtime setup (`_ensure_runtime`: spawn mp + tensor
sharing strategy + structlog).

**SLURM** (`graphids/slurm/`, `graphids/cli/commands.py`) — one Typer
command: `graphids submit --row <json>` submits a single blueprint row
via Parsl `SlurmProvider`. Library entrypoint:
`graphids.slurm.submit.submit_row()`. Reads
`configs/resources/submit_profiles.json` keyed `[mode][cluster][length]`
where each leaf is a `parsl.providers.SlurmProvider` kwargs dict.
Preempt-resume delegated to Lightning's `SLURMEnvironment(auto_requeue=True,
requeue_signal=SIGUSR2)` plugin (wired by `orchestrate._trainer_kwargs`).

The pipeline is strictly one-directional:

```
plan.build(dataset, seed) → list[dict]
    ↓
Plan.model_validate
    ↓
graphids run → JSON array on stdout / file
    ↓
graphids exec --row <json>   (login-node smoke / non-SLURM)
graphids submit --row <json> (SLURM via Parsl; sbatch carries the literal
                              `python -m graphids exec --row '...'` cmd)
    ↓
orchestrate.run_row → trainer.fit / trainer.test / extract / analyze
```

> Authoritative detail: `.claude/rules/config-system.md`,
> `.claude/rules/single-submission-primitive.md`.
