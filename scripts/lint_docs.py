"""Pre-commit gate for the mkdocs site.

Two checks, both cheaper than a full ``mkdocs build --strict`` (~0.3s vs ~6s):

1. **Symbol resolution** — every ``::: target`` directive in ``docs/**/*.md``
   resolves via Griffe (the same static analyzer mkdocstrings uses). Catches
   stale references to renamed/deleted modules at commit time, before CI.
2. **Nav ↔ files symmetry** — every markdown file under ``docs/`` is reachable
   from ``mkdocs.yml`` ``nav:``, and every nav entry resolves to a real file.
   ``--strict`` errors on either side's stragglers; this catches them earlier.

Run directly or via pre-commit. Exit 0 on clean, 1 on any failure with
``file:line`` diagnostics.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MKDOCS_YML = ROOT / "mkdocs.yml"

_DIRECTIVE_RE = re.compile(r"^:::\s+([A-Za-z_][A-Za-z0-9_.]*)\s*$", re.MULTILINE)


def find_directive_targets() -> list[tuple[Path, int, str]]:
    """Return (file, line_no, target) for each ``::: dotted.path`` directive."""
    out: list[tuple[Path, int, str]] = []
    for md in DOCS.rglob("*.md"):
        for n, line in enumerate(md.read_text().splitlines(), start=1):
            m = _DIRECTIVE_RE.match(line)
            if m:
                out.append((md, n, m.group(1)))
    return out


def check_griffe(targets: list[tuple[Path, int, str]]) -> list[str]:
    """Resolve each target via Griffe (static — no imports)."""
    import griffe  # local import: lint may run in a thinner venv

    errors: list[str] = []
    cache: dict[str, bool] = {}
    for path, lineno, target in targets:
        if target in cache:
            ok = cache[target]
        else:
            try:
                griffe.load(target, search_paths=[str(ROOT)])
                ok = True
            except (ImportError, KeyError, ModuleNotFoundError, griffe.GriffeError):
                ok = False
            cache[target] = ok
        if not ok:
            rel = path.relative_to(ROOT)
            errors.append(f"{rel}:{lineno}: ::: {target} — symbol does not exist")
    return errors


def collect_nav_files(nav: list, out: set[str]) -> None:
    """Walk the mkdocs nav recursively, collecting all referenced doc paths."""
    for entry in nav:
        if isinstance(entry, str):
            out.add(entry)
        elif isinstance(entry, dict):
            for v in entry.values():
                if isinstance(v, str):
                    out.add(v)
                elif isinstance(v, list):
                    collect_nav_files(v, out)


class _MkdocsLoader(yaml.SafeLoader):
    """SafeLoader that tolerates mkdocs's ``!!python/name:`` tag."""


_MkdocsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/name:", lambda loader, suffix, node: suffix
)


def check_nav_symmetry() -> list[str]:
    cfg = yaml.load(MKDOCS_YML.read_text(), Loader=_MkdocsLoader)
    nav_paths: set[str] = set()
    collect_nav_files(cfg.get("nav", []), nav_paths)

    # Top-level docs/README.md (GitHub landing page) and drafts/ scratchpad
    # are intentionally not in nav (matching ``exclude_docs`` in mkdocs.yml).
    # Sub-section READMEs (e.g. decisions/README.md) ARE in nav and stay required.
    def _excluded(rel: str) -> bool:
        return rel == "README.md" or rel.startswith("drafts/")

    on_disk = {
        str(p.relative_to(DOCS).as_posix())
        for p in DOCS.rglob("*.md")
        if not _excluded(p.relative_to(DOCS).as_posix())
    }

    errors: list[str] = []
    for missing in sorted(nav_paths - on_disk):
        errors.append(f"mkdocs.yml: nav references {missing} but file does not exist")
    for orphan in sorted(on_disk - nav_paths):
        errors.append(f"docs/{orphan}: file exists but is not in mkdocs.yml nav")
    return errors


def main() -> int:
    errors: list[str] = []
    targets = find_directive_targets()
    errors.extend(check_griffe(targets))
    errors.extend(check_nav_symmetry())

    if errors:
        print("docs lint failed:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    n_targets = len(targets)
    n_files = len(list(DOCS.rglob("*.md")))
    print(f"docs lint OK ({n_targets} directive targets, {n_files} markdown files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
