# GraphIDS Test-Writing Grammar

A test is worth writing only if it encodes something the code itself does
not already guarantee. Answer all three questions before adding one.

## Three questions before writing a test

### 1. Project bug or framework bug?

If the assertion would only fire when Pydantic / Lightning / torch is
broken — **delete**. Framework correctness is tested upstream.

Don't write: `pytest.raises(ValidationError)` for `Literal[...]` /
`frozen=True` / `extra="forbid"`; `pytest.raises(TypeError)` for a
required kwarg being required; `isinstance(cfg.stages, tuple)` after
annotating `stages: tuple[...]`.

Keep custom `@field_validator` logic and cross-field checks Pydantic's
`Literal`/`ge/le` alone don't cover.

### 2. Does the test re-implement the code under test?

If the assertion copies a formula from the implementation, any refactor
cascades into spurious failures. Use **differential** or **property** tests.

```python
# ✗ Mirrors budget.py's internal math
max_nodes = int(free * _SAFETY_MARGIN / effective_bpn)
assert result.budget <= max_nodes

# ✓ Invariant — doesn't reference the formula
assert result.budget >= 1 and result.binding == "memory"

# ✓ Differential — the formula cancels out
dense = _run(edge_p95=210.0).budget
balanced = _run(edge_p95=35 * 4.5).budget
assert dense < balanced

# ✓ Monotonicity — survives any formula refactor
assert larger_gpu_budget >= smaller_gpu_budget
```

### 3. What concrete bug or invariant does this guard?

Every test needs a one-line reason. Put it in the docstring or as a
`# REGRESSION: <context>` / `# INVARIANT: <property>` /
`# CONTRACT: <api guarantee>` comment.

## Markers

Two markers (see `pyproject.toml`):

- `@pytest.mark.slow` — >30s. Deselect with `-m "not slow"`.
- `@pytest.mark.slurm` — needs SLURM: runs `Trainer.fit`, hits CUDA, or
  instantiates models against real datasets. Not for "touches torch" —
  every test does that via `conftest.py`. Reserve for jobs that train.

Over-marking means the test never runs.

## Fixtures and conventions

- **`conftest.py` is lean.** `base_cfg` only holds fields tests actually
  read.
- **`from conftest import ...` is deliberate.** Helpers (`make_graph`,
  `make_batch`, `NUM_IDS`, `IN_CHANNELS`, `EDGE_DIM`) are imported as
  constants. Ruff suppresses `F401` on `tests/*`.
- **Parametrize over contracts, not numerical matrices.** 3×3×3 fine for
  invariants; 72 combos asserting hardcoded numbers is a formula mirror.
- **No silent `pytest.skip()`** for missing fixture files. Either assert
  the file exists or don't parametrize it.

## Organization

- `tests/<layer>/test_<unit>.py` mirrors `graphids/<layer>/<unit>.py`.
- One concept per file.
- Top-level `tests/test_*.py` is for cross-layer integration only.

## Anti-patterns seen in this repo

- Hardcoding VRAM probe measurements or `STAGE_DEPENDENCIES` tuples into a fixture — drifts the moment the producer changes.
- Private Lightning API (`trainer.strategy.connect`) for ckpt roundtrip — use `torch.save(m.state_dict(), ...)` + `load_state_dict`. Canonical: `test_gat.py::TestGATCheckpointRoundtrip`.
