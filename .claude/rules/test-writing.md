# GraphIDS Test-Writing Grammar

A test is worth writing only if it encodes something the code itself does
not already guarantee. Before adding one, answer all three questions.

## Three questions before writing a test

### 1. Project bug or framework bug?

If the assertion would only fire when Pydantic / Lightning / torch / Python
itself is broken — **delete**. Framework correctness is tested upstream.

Patterns deleted from this repo (all framework's job, not ours):
- `pytest.raises(ValidationError)` for `Literal[...]` / `frozen=True` / `extra="forbid"`
- `pytest.raises(TypeError)` for a required kwarg being required
- `isinstance(cfg.stages, tuple)` after annotating `stages: tuple[...]`

Keep only custom `@field_validator` logic and cross-field checks that
Pydantic's `Literal`/`ge/le` alone don't cover.

### 2. Does the test re-implement the code under test?

If the assertion copies the formula from the implementation, any refactor
cascades into 50+ spurious failures. Use **differential** or **property**
tests instead.

Anti-pattern (deleted from `test_budget_matrix.py`):
```python
# ✗ Mirrors budget.py's internal math. Refactoring the formula breaks 72 tests.
max_nodes = int(free * _SAFETY_MARGIN / effective_bpn)
assert result.budget <= max_nodes
```

Replacements:
```python
# ✓ Invariant — doesn't reference the formula
assert result.budget >= 1 and result.binding == "memory"

# ✓ Differential — run twice, compare; the formula cancels out
dense = _run(edge_p95=210.0).budget
balanced = _run(edge_p95=35 * 4.5).budget
assert dense < balanced

# ✓ Monotonicity — property that survives any formula refactor
assert larger_gpu_budget >= smaller_gpu_budget
```

### 3. What concrete bug or invariant does this guard?

Every test needs a one-line reason. If you can't cite one, the test is a
liability — every future refactor will fight it for no gain.

Put the reason in the docstring OR as a `# REGRESSION: <context>` /
`# INVARIANT: <property>` / `# CONTRACT: <api guarantee>` comment.

## Markers

Only two markers are registered. See `pyproject.toml`.

- `@pytest.mark.slow` — >30s. Deselect with `-m "not slow"`.
- `@pytest.mark.slurm` — requires SLURM submission: runs `Trainer.fit`,
  hits CUDA, or instantiates models against real datasets. **Not** for
  "touches torch" — every test touches torch via `conftest.py`, and
  `tests/test_instantiate.py` even builds a full `Trainer` on the
  login node without fitting. Reserve for jobs that actually train.

Over-marking means the test never runs. `@pytest.mark.slow` classes in
`test_gat.py` / `test_vgae.py` exercise training_step on CPU.

## Fixtures and conventions

- **`conftest.py` is lean.** `base_cfg` only holds fields tests actually
  read. If you add a field to a model and tests don't read it from the
  fixture, don't add it to the fixture.
- **`from conftest import ...` is deliberate.** Helpers (`make_graph`,
  `make_batch`, `NUM_IDS`, `IN_CHANNELS`, `EDGE_DIM`) are imported as
  constants. Ruff suppresses `F401` on `tests/*`. Not a mistake.
- **Parametrize over contracts, not numerical matrices.** 3×3×3 is fine
  for invariants; 72 combos each asserting a hardcoded number is a
  formula mirror in disguise.
- **No silent `pytest.skip()`** for missing fixture files. Either assert
  the file exists or don't parametrize it.
- **Don't import `torch` inside test function bodies** unless lazy
  import is genuinely needed (e.g. to isolate a heavy submodule from
  a lightweight test).

## Organization

- `tests/<layer>/test_<unit>.py` mirrors `graphids/<layer>/<unit>.py`.
- One concept per file. The old 717-line `test_overrides.py` (split
  2026-04-04) mixed yaml_utils, recipe_expand, resolver, and KD teacher
  resolution — never again.
- Top-level `tests/test_*.py` is for cross-layer integration only
  (`test_integration.py`, `test_cli_routing_smoke.py`, `test_submit_sh.py`).

## Anti-patterns (seen in this repo, don't repeat)

- Duplicating `topology.py` import-time assertions as pytest functions
  (import-time failure = louder signal than test failure)
- Hardcoding VRAM probe measurements or `STAGE_DEPENDENCIES` tuples
  into a test fixture dict — both drift the moment the producer changes
- Meta-tests like `test_rules_list_has_unique_names` — rule shape is
  ruff/pydantic's job, not pytest's
- Using `trainer.strategy.connect(module)` private Lightning API for
  checkpoint roundtrip — use `torch.save(m.state_dict(), ...)` +
  `load_state_dict(torch.load(..., weights_only=True))` instead
  (canonical shape: `test_gat.py::TestGATCheckpointRoundtrip`)
