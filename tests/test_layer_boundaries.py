"""Layer boundary enforcement tests.

Verifies the 3-layer import hierarchy under graphids/:
    config/    (top)     — never imports from pipeline/ or core/
    pipeline/  (middle)  — never has top-level imports from core/
    core/      (bottom)  — never imports from pipeline/

Uses AST analysis (no runtime imports needed).

Run:  python -m pytest tests/test_layer_boundaries.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = PROJECT_ROOT / "graphids"

CONFIG_DIR = PACKAGE_ROOT / "config"
LAKE_DIR = PACKAGE_ROOT / "lake"
PIPELINE_DIR = PACKAGE_ROOT / "pipeline"
CORE_DIR = PACKAGE_ROOT / "core"


def _collect_python_files(directory: Path) -> list[Path]:
    """Collect all .py files in a directory (recursively)."""
    return sorted(directory.rglob("*.py"))


def _extract_imports(filepath: Path) -> list[tuple[str, bool]]:
    """Extract import targets from a Python file.

    Returns list of (module_name, is_top_level) tuples.
    is_top_level is True if the import is at module scope (not inside a
    function, class, or if TYPE_CHECKING block).
    """
    source = filepath.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Determine if this import is at module scope
            # We consider it top-level if it's a direct child of the Module node
            top_level = _is_top_level_import(tree, node)

            if isinstance(node, ast.Import):
                for alias in node.names:
                    results.append((alias.name, top_level))
            elif node.module:
                results.append((node.module, top_level))
    return results


def _is_top_level_import(tree: ast.Module, target_node: ast.AST) -> bool:
    """Check if an import node is at module scope (not inside function/class/TYPE_CHECKING)."""
    for node in ast.iter_child_nodes(tree):
        if node is target_node:
            return True
        # Check if it's inside an `if TYPE_CHECKING:` block
        if isinstance(node, ast.If):
            test = node.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                if _node_contains(node, target_node):
                    return False
            elif isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                if _node_contains(node, target_node):
                    return False
    return False


def _node_contains(parent: ast.AST, target: ast.AST) -> bool:
    """Check if target node exists anywhere inside parent."""
    for child in ast.walk(parent):
        if child is target:
            return True
    return False


def _subpackage_imported(filepath: Path, top_level_only: bool = False) -> set[str]:
    """Return the set of graphids subpackage names imported by a file.

    Extracts the second-level module name from graphids.X imports.
    E.g. 'from graphids.config import ...' -> {'config'}
         'from graphids.core.models import ...' -> {'core'}
    """
    imports = _extract_imports(filepath)
    modules = set()
    for mod, is_top_level in imports:
        if top_level_only and not is_top_level:
            continue
        parts = mod.split(".")
        if len(parts) >= 2 and parts[0] == "graphids":
            modules.add(parts[1])  # "config", "pipeline", "core"
    return modules


class TestConfigLayerBoundary:
    """config/ must never import from pipeline/ or core/."""

    def test_config_no_pipeline_imports(self):
        violations = []
        for f in _collect_python_files(CONFIG_DIR):
            mods = _subpackage_imported(f)
            if "pipeline" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"config/ imports from pipeline/ (violates layer boundary):\n  "
            + "\n  ".join(violations)
        )

    def test_config_no_core_imports(self):
        violations = []
        for f in _collect_python_files(CONFIG_DIR):
            mods = _subpackage_imported(f)
            if "core" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"config/ imports from core/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )

    def test_config_no_lake_imports(self):
        """config/ must not import from lake/ (lake imports config, not vice versa)."""
        violations = []
        for f in _collect_python_files(CONFIG_DIR):
            mods = _subpackage_imported(f)
            if "lake" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"config/ imports from lake/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )


class TestLakeLayerBoundary:
    """lake/ must never import from pipeline/ or core/ (same layer as config/)."""

    def test_lake_no_pipeline_imports(self):
        violations = []
        for f in _collect_python_files(LAKE_DIR):
            mods = _subpackage_imported(f)
            if "pipeline" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"lake/ imports from pipeline/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )

    def test_lake_no_core_imports(self):
        violations = []
        for f in _collect_python_files(LAKE_DIR):
            mods = _subpackage_imported(f)
            if "core" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"lake/ imports from core/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )


class TestCoreLayerBoundary:
    """core/ must never import from pipeline/."""

    def test_core_no_pipeline_imports(self):
        violations = []
        for f in _collect_python_files(CORE_DIR):
            mods = _subpackage_imported(f)
            if "pipeline" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"core/ imports from pipeline/ (violates layer boundary):\n  " + "\n  ".join(violations)
        )


class TestPipelineLayerBoundary:
    """pipeline/ must not have top-level imports from core/ (lazy/function-local OK)."""

    def test_pipeline_no_toplevel_core_imports(self):
        violations = []
        for f in _collect_python_files(PIPELINE_DIR):
            mods = _subpackage_imported(f, top_level_only=True)
            if "core" in mods:
                violations.append(str(f.relative_to(PROJECT_ROOT)))
        assert not violations, (
            f"pipeline/ has top-level imports from core/ (should be lazy/function-local):\n  "
            + "\n  ".join(violations)
        )


class TestGatewayEnforcement:
    """External callers must use package gateways, not deep submodule imports."""

    def _extract_deep_core_imports(self, filepath: Path) -> list[str]:
        """Find imports like 'from graphids.core.X.Y import ...' (3+ levels deep)."""
        imports = _extract_imports(filepath)
        deep = []
        for mod, _is_top_level in imports:
            parts = mod.split(".")
            # graphids.core.X.Y = 4+ parts, meaning depth > 2 under graphids.core
            if len(parts) >= 4 and parts[0] == "graphids" and parts[1] == "core":
                deep.append(mod)
        return deep

    def test_pipeline_uses_core_gateway(self):
        """Pipeline layer should import from graphids.core or graphids.core.X,
        not graphids.core.X.Y (with exceptions for backward-compat re-exports)."""
        # Allow these specific deep imports (lazy, inside functions)
        allowed_deep = {
            "graphids.core.data",  # Dataset loading gateway
            "graphids.core.models.registry",  # Registry functions
            "graphids.core.models.dqn",  # DQN model classes (fusion/eval/serve)
            "graphids.core.models.vgae",  # VGAE Lightning module
            "graphids.core.models.gat",  # GAT Lightning module
            "graphids.core.models.temporal",  # Temporal model
            "graphids.core.preprocessing.dataset",  # CollatedGraphDataset
            "graphids.core.preprocessing.parallel",  # process_dataset
            "graphids.core.preprocessing.temporal",  # TemporalGrouper
            "graphids.core.preprocessing.vocabulary",  # EntityVocabulary
            "graphids.core.preprocessing.adapters.can_bus",  # CANBusAdapter
            "graphids.core.training.datamodules",  # Backward-compat (re-exports)
        }
        violations = []
        for f in _collect_python_files(PIPELINE_DIR):
            for mod in self._extract_deep_core_imports(f):
                if mod not in allowed_deep:
                    violations.append(f"{f.relative_to(PROJECT_ROOT)}: {mod}")
        assert not violations, (
            "Pipeline imports from deep core submodules (use gateway instead):\n  "
            + "\n  ".join(violations)
        )
