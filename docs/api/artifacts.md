# Artifacts

Per-checkpoint artifact generation: embeddings, GAT attention weights,
teacher↔student CKA, loss-landscape grids, fusion-policy traces. Driven
by `AnalysisConfig` and dispatched directly through
`graphids.exp.runtime.run_stage` → `Analyzer(spec).run()`.

Distinct from [`graphids.analysis`](#graphids.analysis), which owns
cross-run statistical comparison from the MLflow catalog (no torch,
login-safe).

## Layout

```
graphids/core/artifacts/
├── analyzer.py      orchestrates load → compute → save loop, writes manifest
├── _dispatch.py     ARTIFACTS table — each row is the only place compute + I/O meet
├── compute.py       pure compute fns + frozen result dataclasses (no fs)
├── io.py            every read (val data, teacher ckpt, fusion eval) + every write
└── __init__.py
```

The compute / I/O split is structural: every `compute_*` function takes
pre-loaded models and pre-built `val_data`, returns a frozen dataclass,
and never touches the filesystem. `io.save_*` consumes those dataclasses
and writes; `io.load_*` reads. The dispatch table is the single seam
between the two — adding an artifact means one new row in `ARTIFACTS`,
one new compute fn, and one new save fn.

## Adding an artifact

1. Add `compute_X(...) -> XResult` (frozen dataclass) to `compute.py`.
2. Add `save_X(out, result)` to `io.py`.
3. Add an `Artifact("X", "x.npz", frozenset({...}), _run_X)` row to
   `_dispatch.ARTIFACTS` and a `_run_X` glue fn that wires load →
   compute → save.
4. Add a toggle field on `AnalysisConfig` (default off, or default-on for the
   model types in `applies_to` via `default_toggles_for`).

`expected_outputs(spec)` and `Analyzer.run()` derive from `ARTIFACTS`
automatically — no parallel declaration to update.

## Reuse with training/eval

`io.load_val_data` goes through `CANBusSource` → `state.get_or_build`
— the same path
[`GraphDataModule.setup`](data.md#graphids.core.data.datamodule.graph)
takes during training. `val_fraction`, scaler strategy, and cache digest
live on the source dataclass; the analyzer picks up changes there
automatically with no parallel declaration.

`io.load_teacher` and the student ckpt load in `Analyzer.run` both go
through
[`safe_load_checkpoint`](models.md#graphids.core.models.base.safe_load_checkpoint)
— the canonical "ckpt → module" registry. `io.load_fusion_eval` wraps
the same `FusionDataModule` training/eval uses.

## Manifest sidecar

`Analyzer.run()` writes `analysis_manifest.json` next to the artifacts:
the rendered analysis identity, expected outputs (derived from
`expected_outputs(spec)`), and which actually exist on disk after the
run. Useful as provenance when an `analyze` run was submitted via SLURM
and the output dir is the only artifact left.

## `graphids.core.artifacts`

::: graphids.core.artifacts
    options:
      members: true
      show_submodules: true
