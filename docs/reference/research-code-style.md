# Research code style

GraphIDS favors code a researcher can read from configuration through data,
model, loss, and evaluation. Use the existing Python, PyTorch, Lightning,
Pydantic, and Ruff conventions; keep the algorithm visible.

## Module organization

- Use shallow domain packages: `data`, `models`, `exp`, and `cli`. Shared
  evaluation, metrics, and callbacks live directly under `graphids`.
- Keep data/model configuration in the owning package's `config.py`, with
  descriptive `Config` class names. Experiment composition belongs in `exp`.
- Import from the owning module. Avoid catch-all facades, generic config
  aliases, and wrapper factories that only call a config constructor.

## Interfaces and validation

- Validate configuration with the domain-owned Pydantic models. Direct constructors
  and checkpoint reconstruction use those same definitions. Do not maintain a
  second set of handwritten range and cross-field checks.
- Validate external recordings, caches, checkpoints, and stream transitions at
  their owning boundaries. Internal helpers trust the resulting objects.
- Retain checks for provenance, temporal continuity, scored-row coverage,
  checkpoint compatibility, and capacities that could otherwise lose messages.
  Tests cannot detect changed inputs during a future experiment.
- Prefer explicit arguments, stable return types, and direct method calls.
  Normalize supported external forms once, rather than probing for methods or
  accepting several internal representations throughout a pipeline.
- Optional hooks belong at integration boundaries. For example, a custom
  class-path model may omit event-data preparation; event-model internals can
  rely on a prepared event datamodule.

## Readable scientific logic

- Separate file parsing and artifact publication from numerical calculations.
  Use functions on tensors/arrays and existing dataclasses for research state.
- Give functions one clear job. Prefer a short sequence of named operations to
  nested conditionals, compressed expressions, or a configurable framework.
- Annotate interfaces and explain tensor shapes, scientific assumptions, and
  non-obvious mathematics. Use paper notation when it clarifies an equation.
- Initialize optional state explicitly instead of repeatedly using `getattr`
  with fallback values. Avoid adapters and aliases without an active consumer.
- Let unexpected exceptions propagate with their original traceback. Do not
  catch a builder's `TypeError` and retry with different arguments.
- Catch errors for actual recovery, cleanup, useful input context, or optional
  telemetry isolation. Keep the protected block small. Use context managers or
  `finally` for cleanup; do not replace required validation with `assert`.
- Preserve spawn-based CUDA multiprocessing, model training mode, numerical
  stability, and bounded kernel workspaces when simplifying control flow.

## Tests and tools

- Test distinct scientific or workflow behavior. Keep independent numerical
  oracles, gradient comparisons, causality, masking, coverage, and resume tests.
- Remove tests of upstream framework guarantees and obsolete interfaces. Reduce
  duplicate parameter combinations; do not combine unrelated behaviors just to
  lower the reported count.
- Share synthetic inputs through `tests/event_fixtures.py`; never import helpers
  from another test module. Mock external services at their boundaries.
- Synthetic CPU cases are unmarked. Mark training cases `slurm`, and CUDA cases
  both `gpu` and `slurm`. On HPC, run every pytest command inside an allocation.
  Use `python -m pytest -m 'not gpu and not slurm'` for the CPU development suite.
- Run `ruff check graphids tests scripts` and `ruff format --check` on changed
  Python files. Avoid blanket lint suppressions and complexity/count quotas.

## Sources

These conventions adapt the Scientific Python guides on
[design](https://learn.scientific-python.org/development/principles/design/) and
[testing](https://learn.scientific-python.org/development/principles/testing/),
and the [Google Python exception guidance](https://google.github.io/styleguide/pyguide.html#24-exceptions).
They are project conventions, not a claim that ML research has one universal style.
