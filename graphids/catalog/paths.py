"""Run-dir parsing and identity derivation.

The ablation tree produces run_dirs of the shape

    .../<dataset>/ablations/<group>/<variant>/seed_<N>

(see ``configs/ablations/_paths.libsonnet``). ``parse_run_dir`` walks the
tail of the path; no regex, no anchor on ``lake_root`` (so a test
``tmp_path`` parses the same way a GPFS path does).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunIdentity:
    """The (group, variant, dataset, seed) tuple that identifies a run."""

    group: str
    variant: str
    dataset: str
    seed: int


def parse_run_dir(run_dir: Path) -> RunIdentity | None:
    """Return the identity tuple for an ablation run_dir, or ``None`` if the
    path doesn't match ``.../<dataset>/ablations/<group>/<variant>/seed_<N>``.

    No exception on mismatch — callers use ``None`` as the skip signal
    (bare ``stages/*.jsonnet`` runs, dev smokes).
    """
    parts = Path(run_dir).parts
    if len(parts) < 5:
        return None
    seed_part, variant, group, ablations_marker, dataset = (
        parts[-1],
        parts[-2],
        parts[-3],
        parts[-4],
        parts[-5],
    )
    if ablations_marker != "ablations" or not seed_part.startswith("seed_"):
        return None
    try:
        seed = int(seed_part.removeprefix("seed_"))
    except ValueError:
        return None
    return RunIdentity(group=group, variant=variant, dataset=dataset, seed=seed)


def run_id_for(identity: RunIdentity, cluster: str | None = None) -> str:
    """Build the stable run_id: ``{group}_{variant}_{dataset}_seed{N}[_{cluster}]``.

    Cluster suffix is appended only when provided — keeps backward
    compatibility with already-cataloged runs that had no cluster set.
    """
    base = f"{identity.group}_{identity.variant}_{identity.dataset}_seed{identity.seed}"
    return f"{base}_{cluster}" if cluster else base
