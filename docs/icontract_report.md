# iContract Library Report

## What It Is

[icontract](https://github.com/Parquery/icontract) (v2.7.3, MIT license) implements Design by Contract (DbC) for Python 3.6--3.13 via decorators. It provides preconditions (`@require`), postconditions (`@ensure`), and class invariants (`@invariant`) with auto-generated violation messages that include the condition source code and all referenced variable values at breach time. Its distinguishing feature over alternatives (dpcontracts, deal, PyContracts) is correct contract inheritance following the Liskov Substitution Principle.

## Core API

### Preconditions -- `@icontract.require(condition, description?, enabled?, error?)`

Checked before function body. Condition is a lambda/callable accepting a subset of the function's args.

```python
@icontract.require(lambda x: x > 0, "x must be positive")
def train_step(x: Tensor) -> Tensor: ...
```

Violation raises `ViolationError` with: file location, condition source, description, and all variable values.

### Postconditions -- `@icontract.ensure(condition, description?, enabled?, error?)`

Checked after return. Condition receives `result` plus any original args. Supports async functions.

```python
@icontract.ensure(lambda result: result.shape[0] > 0)
def collate(batch: list[Data]) -> Batch: ...
```

### Class Invariants -- `@icontract.invariant(condition, check_on?)`

Checked after `__init__`, before/after public methods, and after dunder methods. NOT checked on private (`_`-prefixed) methods, `__repr__`, `__getattribute__`, classmethods, or `__new__`.

`check_on` parameter controls granularity: `InvariantCheckEvent.CALL` (default), `.SETATTR`, or `.ALL`. SETATTR mode catches `self.x = val` but not mutations through references (`self.lst.append()`).

### Snapshots -- `@icontract.snapshot(capture, name?)`

Captures argument state before execution for use in postconditions via the `OLD` parameter.

```python
@icontract.snapshot(lambda lst: lst[:])
@icontract.ensure(lambda OLD, lst, value: lst == OLD.lst + [value])
def append_item(lst: list[int], value: int) -> None:
    lst.append(value)
```

Named snapshots: `@icontract.snapshot(lambda lst: len(lst), name="len_lst")` -> access as `OLD.len_lst`. Single-arg captures default to the argument name. Snapshots are inherited from base classes.

### Custom Errors -- `error` parameter

Three forms: (1) callable returning exception (cheapest -- skips value tracing), (2) exception class, (3) exception instance.

```python
@icontract.require(lambda x: x > 0, error=lambda x: ValueError(f"got {x}"))
```

### Inheritance -- `DBC` base class / `DBCMeta` metaclass

Classes MUST inherit from `icontract.DBC` (or use `DBCMeta`) for contract inheritance. Without it, contracts from parent classes leak silently.

- **Preconditions weaken**: child preconditions OR'd with parent's (broader inputs accepted)
- **Postconditions strengthen**: child postconditions AND'd with parent's (stricter outputs required)
- **Invariants strengthen**: child invariants added to parent's

`DBCMeta` inherits from `abc.ABCMeta`, so it composes with abstract base classes.

### Toggling

1. **`enabled` parameter**: `@icontract.require(..., enabled=False)` disables the check entirely.
2. **`icontract.SLOW` flag**: Reflects `ICONTRACT_SLOW` env var. Use `enabled=icontract.SLOW` for expensive checks that should only run in dev/CI.
3. **`python -O`**: Disables all contracts globally (respects `__debug__`).

### Async Support

Sync contracts work on coroutines directly. Async conditions require async condition functions. Invariants cannot be async (they wrap synchronous dunders). If a condition returns a coroutine, icontract awaits it, but explicit error messages are required (no re-computation for diagnostics).

### Representation -- `aRepr`

Global `icontract.aRepr` controls how values render in violation messages. Based on `reprlib.Repr` with configurable truncation limits.

## Benchmarks (Intel Xeon E-2276M, Python 3.9.9)

| Operation | icontract | dpcontracts | deal | Inline baseline |
|---|---|---|---|---|
| `@require` | ~3.9 us | ~53.9 us | ~4.2 us | ~0.15 us |
| `@ensure` | ~4.4 us | ~52.5 us | ~1.0 us | ~0.15 us |
| `@invariant` (init) | ~1.5 us | ~0.5 us | ~1.7 us | ~0.28 us |
| `@invariant` (call) | ~2.0 us | ~0.5 us | ~4.7 us | ~0.23 us |

icontract is ~25x slower than inline checks but in the microsecond range. dpcontracts and deal lack inheritance support, which explains their speed in invariant cases.

## Integration Points for GraphIDS

### Config Pipeline (Pydantic already handles this)

icontract would overlap with `validate_config` / `ValidatedConfig` Pydantic validators. Pydantic is better here: it provides typed schemas, serialization, and already gates the entire config path. No benefit to adding `@require` on top.

### Model Forward Pass / Training Loop

Contracts on `forward()`, `training_step()`, or data transforms would fire every batch. At ~4 us per contract and ~100-1000 batches/epoch, overhead is 0.4--4 ms/epoch total -- negligible vs GPU compute. Useful candidates:

- **Tensor shape invariants**: `@require(lambda x: x.dim() == 2)` on encoder input
- **No-NaN postconditions**: `@ensure(lambda result: not result.isnan().any())` on loss computation (use `enabled=icontract.SLOW` since `.isnan().any()` has GPU sync cost)
- **Data pipeline**: `@invariant` on `Data` wrapper classes to assert `edge_index.max() < num_nodes`

### Critical Constraints Enforcement

The project's `critical-constraints.md` rules (clamp skewness to +/-10, clone before `.to()`, spawn not fork) could be expressed as contracts, but most are already enforced by code structure or Pydantic. Contracts would add a second enforcement layer with better diagnostics on violation.

### Knowledge Distillation Auxiliaries

`@ensure` on teacher-student score alignment (e.g., `result.shape == teacher_output.shape`) would catch wiring bugs early with informative messages.

## Limitations and Trade-offs

1. **~25x overhead vs inline checks.** Microseconds per contract, but accumulates with many contracts on hot paths. GPU-bound ML training is unlikely to notice; CPU-bound data preprocessing might.
2. **Condition functions must be side-effect-free.** icontract re-executes conditions via AST traversal for error message generation. Any side effects run twice on violation.
3. **Lambda limitations.** Python lambdas are single-expression. Complex invariants need named functions, which lose the inline readability advantage.
4. **No mutation detection through references.** `@invariant` with `SETATTR` catches `self.x = 5` but not `self.x.append(5)`.
5. **Decorator stacking order matters.** `@require` must be below `@ensure` in decorator order (closer to the function). Multiple `@require` decorators are AND'd.
6. **DBC inheritance is opt-in.** Forgetting to inherit from `DBC` silently leaks parent contracts -- a subtle correctness bug.
7. **Mypy interaction.** `@require` historically stripped type annotations for mypy (issue #93). Check current status before adopting.
8. **Overlap with Pydantic.** For config validation, Pydantic `@field_validator` / `@model_validator` already provides typed, serializable validation with better error messages. icontract adds value only where Pydantic doesn't reach (runtime tensor checks, function I/O contracts).
9. **No native PyTorch tensor support.** Tensor comparisons return tensors, not bools. Conditions must call `.item()`, `.all()`, or `.any()` explicitly, and GPU tensors require sync.
10. **Ecosystem size.** ~400 GitHub stars. Maintained but niche. The hypothesis integration (`icontract-hypothesis`) is the most compelling ecosystem piece for testing.

## Verdict

icontract is well-designed for pure-Python DbC with excellent error messages and proper inheritance semantics. For GraphIDS specifically, the overlap with Pydantic config validation is near-total for the config pipeline. The strongest use case would be `SLOW`-gated tensor shape/NaN contracts on model I/O during development, where Pydantic cannot reach and the auto-generated violation messages (showing actual tensor shapes and values) would speed up debugging. Adoption cost is low (decorator-only, no framework changes), but the benefit is marginal given that most invariants are already enforced by existing mechanisms.
