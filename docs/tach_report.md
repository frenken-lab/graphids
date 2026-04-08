# Tach: Python Module Boundary Enforcement

**Repo:** github.com/gauge-sh/tach (2.7k stars, Rust core, pip-installable)
**Docs:** docs.gauge.sh | **License:** MIT | **Analysis:** static (AST-based, zero runtime cost)

## What It Is

Tach enforces Python module boundaries via static import analysis. You declare which modules exist, what they may depend on, and what they expose publicly. `tach check` scans the AST and fails on violations. No runtime overhead -- it never executes your code.

## Core Concepts

1. **Modules** -- named Python packages/files you want to govern (e.g., `graphids.config`, `graphids.core`).
2. **Dependencies** (`depends_on`) -- explicit allowlist of which modules a given module may import from.
3. **Interfaces** (`expose`) -- regex patterns defining a module's public API; imports of non-exposed symbols fail.
4. **Layers** -- ordered list (high-to-low); higher layers may import lower ones, not vice versa.

## Configuration (`tach.toml`)

```toml
layers = ["cli", "orchestrate", "core", "config"]
source_roots = ["."]
exact = true                          # fail on unused declared deps
forbid_circular_dependencies = true
ignore_type_checking_imports = true   # default: skip TYPE_CHECKING blocks

[[modules]]
path = "graphids.config"
depends_on = []                       # no internal deps allowed
layer = "config"

[[modules]]
path = "graphids.core"
depends_on = ["graphids.config"]
layer = "core"

[[modules]]
path = "graphids.orchestrate"
depends_on = ["graphids.config", "graphids.slurm"]
layer = "orchestrate"
# NOTE: no "graphids.core" -- this is the boundary we enforce

[[interfaces]]
expose = ["render_config", "validate_config", "ValidatedConfig"]
from = ["graphids.config"]
```

**Key fields per module:** `path`, `depends_on`, `cannot_depend_on`, `layer`, `visibility` (who can import this module), `utility` (importable by all), `unchecked` (skip checking -- for incremental adoption).

**Strictness knobs:** `exact` (unused deps = error), `forbid_circular_dependencies`, `root_module` (`ignore`/`allow`/`forbid` for uncovered code), `layers_explicit_depends_on` (require explicit cross-layer deps).

## Enforcement Mechanism

- **Pure static analysis.** Parses the AST to find `import` / `from ... import` statements. No runtime hooks, no monkey-patching, no `sys.meta_path` manipulation.
- **TYPE_CHECKING blocks** are ignored by default (`ignore_type_checking_imports = true`). Togglable.
- **Dynamic imports** (`importlib.import_module("foo")`, string-based paths) are invisible to tach. Use `# tach-ignore` for these.
- **Conditional imports** (inside `if`/`try` blocks) ARE checked -- the AST sees them regardless of runtime reachability. Only `TYPE_CHECKING` gets special treatment.

## Handling Exceptions

Inline suppression with optional reason:
```python
from graphids.core.models import GAT  # tach-ignore(lazy import for dagster definition-time purity)
```
Selective suppression for multi-import lines:
```python
from graphids.core import train, GAT  # tach-ignore GAT
```
Rules can require reasons: `[rules] require_ignore_directive_reasons = "error"`.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `tach init` | Guided interactive setup |
| `tach mod` | Interactive module boundary editor (TUI) |
| `tach sync` | Auto-update `depends_on` from actual imports |
| `tach check` | Validate boundaries (exit 1 on violation) |
| `tach check --exact` | Also fail on unused declared deps |
| `tach check-external` | Validate third-party imports vs pyproject.toml |
| `tach show [--web\|--mermaid]` | Dependency graph visualization |
| `tach report <path>` | Show deps + usages for a module |
| `tach map` | JSON file-level dependency map |
| `tach test` | Run only tests affected by changed modules |
| `tach install pre-commit` | Git pre-commit hook |

## CI/CD Integration

- **Pre-commit hook:** `tach install pre-commit` writes to `.git/hooks/pre-commit`, or use the pre-commit framework repo (`gauge-sh/tach-pre-commit`).
- **CI:** `tach check` exits non-zero on violation. Add to GitHub Actions / any CI pipeline.
- **VS Code:** Extension `detachhead.dtach` provides inline violation highlighting.
- **Affected tests:** `tach test --base main` runs only tests whose dependency graph touches changed files. Supports pytest plugin (`--tach` flag).

## Concrete Use Cases for GraphIDS

1. **Prevent `orchestrate` -> `core` imports at definition time.**
   The dagster module (`graphids.orchestrate.dagster`) runs on the login node. Importing torch/Lightning at definition time crashes. Tach can enforce `graphids.orchestrate` never depends on `graphids.core` (which contains models/training). Lazy imports inside `@asset` functions would need `# tach-ignore` with a reason.

2. **Enforce `config/` never imports torch.**
   Mark `graphids.config` as a bottom layer with `depends_on = []`. Any torch import (direct or transitive via `graphids.core`) fails `tach check`.

3. **Layered architecture enforcement.**
   `layers = ["cli", "orchestrate", "core", "config"]` -- CLI can import anything below; config cannot import anything above. Matches the existing layered design.

4. **Interface enforcement for config public API.**
   Expose only `render_config`, `validate_config`, `ValidatedConfig` from `graphids.config`. Prevents other modules from reaching into `config/jsonnet.py` internals.

5. **Deprecation tracking.**
   Mark known violations as `deprecated = true` in `depends_on` -- tach warns but doesn't fail. Provides a migration path for cleaning up existing boundary violations.

6. **Incremental adoption.**
   Mark `graphids.plots` or unstable modules as `unchecked = true` to skip enforcement while tightening boundaries on critical modules first.

## Comparison with import-linter

| Aspect | tach | import-linter |
|--------|------|---------------|
| Language | Rust (fast) | Python |
| Config format | `tach.toml` | `.importlinter` / `setup.cfg` |
| Contract types | deps + interfaces + layers + visibility | layers, independence, forbidden |
| Public interface enforcement | Yes (expose patterns) | No |
| Visualization | DOT, Mermaid, web viewer | Browser-based explorer |
| Affected test detection | Yes (`tach test`) | No |
| Interactive setup | Yes (TUI via `tach mod`) | No |
| Deprecation workflow | Yes (warn-not-fail) | No |
| Incremental adoption | `unchecked` modules, `root_module` | Manual exclusion |
| External dep checking | Yes (`tach check-external`) | No |
| Pre-commit integration | Built-in + framework hook | Manual |
| Community | 2.7k stars, active | 991 stars, mature |

**import-linter** is simpler and sufficient for basic layer/independence contracts. **tach** is strictly more capable: interface enforcement, deprecation workflows, affected-test detection, and Rust-speed scanning make it the better fit for a project that already cares about module boundaries.

## Limitations and Trade-offs

1. **Static only.** `importlib.import_module()` calls are invisible. GraphIDS uses `importlib` in `instantiate.py` for class_path resolution -- these would need `# tach-ignore`.
2. **Conditional imports are checked.** A `try: import torch` inside a function body is flagged even if it's a graceful fallback. Must use `# tach-ignore`.
3. **No transitive dependency analysis for boundaries.** If `A -> B -> C` and you forbid `A -> C`, tach only checks direct imports. B importing C and A importing B is fine. The boundary is per-import-statement, not per-reachability.
4. **Config maintenance.** Every new submodule needs a `tach.toml` entry (or use glob patterns). `tach sync` helps but adds discovered deps rather than enforcing intended ones.
5. **No runtime enforcement.** Won't catch monkey-patched imports or plugin-loaded modules. Purely a development/CI gate.
6. **Dynamic import patterns in ML.** PyTorch/Lightning use dynamic class loading extensively. The `# tach-ignore` escape hatch handles this, but noisy configs are possible if overused.
