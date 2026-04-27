"""Resolve ``--depends-on <variant>[:<seed>]`` into upstream-ckpt TLAs via MLflow tags.

The variant→TLA mapping (:data:`DEPENDS_ON_TLA`) is the single source of
truth for the CLI's ``--depends-on`` flag (atomic submissions). Plan
nodes don't go through this — ``configs/plans/*.jsonnet`` declares
upstream ckpts directly via ``std.native('paths.best_ckpt')(...)``.
Each entry maps a *producer* variant name (e.g. ``"vgae"``) to the TLA
name a *consumer* preset uses to receive that producer's ckpt path
(e.g. ``"vgae_ckpt_path"``).

The resolver queries MLflow for the latest FINISHED fit run matching
``(dataset, variant, seed)``, reads its ``graphids.run_dir`` tag, and
returns ``<run_dir>/checkpoints/best_model.ckpt``. All resolution
failures raise :class:`DependencyResolutionError` with an actionable
message — the CLI boundary surfaces the message verbatim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import typer

_CHUNK_RE = re.compile(r"^([A-Za-z_][\w-]*)(?::(\d+))?$")

# Producer variant → consumer-preset TLA name. Add entries when a new
# variant is exposed as a teacher for a downstream preset.
DEPENDS_ON_TLA: dict[str, str] = {
    "vgae": "vgae_ckpt_path",
    "focal": "gat_ckpt_path",
}


class DependencyResolutionError(Exception):
    """``--depends-on`` could not be resolved (not found, missing tag, missing ckpt)."""


@dataclass(frozen=True)
class DependencySpec:
    """One ``<variant>[:<seed>]`` entry parsed from ``--depends-on``."""

    variant: str
    seed: int


def parse_depends_on(raw: str, default_seed: int | None) -> list[DependencySpec]:
    """Parse ``"<v>[:<s>][,<v>[:<s>]]"`` into specs. ``default_seed`` fills omitted seeds."""
    specs: list[DependencySpec] = []
    for chunk in (c.strip() for c in raw.split(",") if c.strip()):
        m = _CHUNK_RE.match(chunk)
        if not m:
            raise typer.BadParameter(f"--depends-on entry {chunk!r}: expected '<variant>[:<seed>]'")
        variant, seed_str = m.group(1), m.group(2)
        if seed_str is None and default_seed is None:
            raise typer.BadParameter(
                f"--depends-on entry {variant!r} has no seed; pass --seed N or use '{variant}:N'"
            )
        specs.append(
            DependencySpec(variant=variant, seed=int(seed_str) if seed_str else default_seed)
        )
    return specs


def resolve_dependency(dep: DependencySpec, dataset: str) -> Path:
    """Look up the latest FINISHED fit run for ``dep`` and return the ckpt path.

    Raises :class:`DependencyResolutionError` with an actionable message
    on any of: no matching run, missing ``graphids.run_dir`` tag (pre-2026-04
    rows), or ckpt deleted from disk. MLflow transient errors propagate.
    """
    from graphids._mlflow import latest_run

    row = latest_run(
        dataset=dataset, variant=dep.variant, seed=dep.seed, phase="fit", status="FINISHED"
    )
    if row is None:
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: no FINISHED fit run in "
            f"MLflow for dataset={dataset}. Submit it first: "
            f"`graphids submit configs/ablations/<group>/{dep.variant}.jsonnet "
            f"--dataset {dataset} --seed {dep.seed}`"
        )
    run_dir_tag = row.get("tags.graphids.run_dir")
    if not run_dir_tag or not isinstance(run_dir_tag, str):
        tla = DEPENDS_ON_TLA.get(dep.variant, "<role>_ckpt_path")
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: MLflow run "
            f"{row.get('run_id', '<unknown>')} has no graphids.run_dir tag "
            f"(probably a pre-2026-04 run). Re-fit or pass --tla {tla}=<path> manually."
        )

    ckpt = Path(run_dir_tag) / "checkpoints" / "best_model.ckpt"
    if not ckpt.exists():
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: ckpt missing on disk: "
            f"{ckpt}. Run was FINISHED but checkpoint was deleted/moved."
        )
    return ckpt


def build_dependency_tlas(specs: list[DependencySpec], dataset: str) -> list[tuple[str, str]]:
    """Resolve each spec via :func:`resolve_dependency` and return TLA pairs.

    Variants not in :data:`DEPENDS_ON_TLA` raise :class:`typer.BadParameter`.
    """
    tlas: list[tuple[str, str]] = []
    for spec in specs:
        if spec.variant not in DEPENDS_ON_TLA:
            raise typer.BadParameter(
                f"--depends-on {spec.variant}: not in dependency registry. "
                f"Register it in graphids/slurm/dependencies.py:DEPENDS_ON_TLA "
                f"or pass --tla manually. Known: {sorted(DEPENDS_ON_TLA)}"
            )
        tla_name = DEPENDS_ON_TLA[spec.variant]
        ckpt = resolve_dependency(spec, dataset)
        tlas.append((tla_name, str(ckpt)))
    return tlas
