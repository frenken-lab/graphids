# GraphIDS tests

Follow `docs/reference/research-code-style.md`.

## What to test

- Name the scientific invariant, regression, or workflow contract each test guards.
- Preserve independent output/gradient oracles, causal isolation, masking,
  complete scored-row coverage, objective normalization, and checkpoint/RNG resume.
- Test custom cross-field configuration logic at representative accepted/rejected
  boundaries. Do not retest Pydantic literals, frozen models, or required arguments.
- Avoid copying implementation formulas and exhaustive configuration grids that
  only prove values are stored. Prefer differential and property tests.
- Remove a test when its only consumer or obsolete interface is removed. Keep one
  readable test per distinct behavior rather than large catch-all tests.

## Execution

- On HPC, all pytest commands run in a SLURM allocation, never a login node.
- Unmarked tests use synthetic CPU inputs. CPU-only selection:
  `python -m pytest -m 'not gpu and not slurm'`.
- `slurm` marks training and compute-dependent integration cases; `gpu` plus
  `slurm` marks CUDA cases. Mark individual parameters when only some need CUDA.
- `slow` marks cases over 30 seconds. Do not mark tests merely for importing torch.

## Organization

- Keep `conftest.py` limited to test execution settings.
- Shared synthetic recordings/models live in `tests/event_fixtures.py`, and the
  independent numerical oracle lives in `tests/event_reference.py`.
- Never import helpers from a test module. Mock external services at boundaries.
- Unit tests follow package responsibilities; top-level tests cover cross-layer
  event behavior. Ruff checks unused test imports without a blanket exemption.
