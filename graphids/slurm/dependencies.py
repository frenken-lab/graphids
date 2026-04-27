"""Resolve ``--depends-on <variant>[:<seed>]`` into ckpt TLAs and/or afterok jids.

One CLI primitive (`--depends-on`) covers two cases via MLflow lookup:

- Upstream **FINISHED** → inject ``<role>_ckpt_path`` TLA only. Downstream
  reads the existing checkpoint; no SLURM dependency needed.
- Upstream **RUNNING** → inject the TLA (path is deterministic from
  ``run_dir``) AND add the upstream's ``slurm.slurm_job_id`` as an afterok
  dep. Downstream queues now and waits on the running upstream.
- Upstream **missing / FAILED / KILLED** → hard error. User re-submits
  upstream, then retries.

Both cases share the same flag and same registry. No second knob, no
``--dep`` jid plumbing — see ``.claude/rules/single-submission-primitive.md``.
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
    """``--depends-on`` could not be resolved (missing run, bad state, missing tag)."""


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


def resolve_dependency(dep: DependencySpec, dataset: str) -> tuple[Path, int | None]:
    """Look up upstream MLflow row; dispatch by status.

    Returns ``(ckpt_path, afterok_jid)``. ``ckpt_path`` is always set
    (read from ``graphids.run_dir`` tag — works for both FINISHED and
    RUNNING since the tag is stamped at fit-start). ``afterok_jid`` is
    set only when the upstream is currently RUNNING (read from the
    ``slurm.slurm_job_id`` tag). Both terminal-bad statuses (FAILED,
    KILLED) and missing rows raise :class:`DependencyResolutionError`.
    """
    from graphids._mlflow import latest_run

    row = latest_run(dataset=dataset, variant=dep.variant, seed=dep.seed, phase="fit")
    if row is None:
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: no fit run in MLflow for "
            f"dataset={dataset}. Submit it first: "
            f"`graphids submit configs/ablations/<group>/{dep.variant}.jsonnet "
            f"--dataset {dataset} --seed {dep.seed}`"
        )
    status = str(row.get("status", ""))
    if status not in ("FINISHED", "RUNNING"):
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: upstream status is {status!r}. "
            f"Need FINISHED (use existing ckpt) or RUNNING (afterok dep) — "
            f"re-submit it."
        )
    run_dir_tag = row.get("tags.graphids.run_dir")
    if not run_dir_tag or not isinstance(run_dir_tag, str):
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: MLflow run "
            f"{row.get('run_id', '<unknown>')} has no graphids.run_dir tag."
        )
    ckpt = Path(run_dir_tag) / "checkpoints" / "best_model.ckpt"
    if status == "FINISHED":
        if not ckpt.exists():
            raise DependencyResolutionError(
                f"--depends-on {dep.variant}:{dep.seed}: ckpt missing on disk: "
                f"{ckpt}. Run was FINISHED but checkpoint was deleted/moved."
            )
        return ckpt, None
    # RUNNING — ckpt may not exist yet; afterok gates the downstream submit.
    jid_tag = row.get("tags.slurm.slurm_job_id")
    if not jid_tag or not isinstance(jid_tag, str) or not jid_tag.isdigit():
        raise DependencyResolutionError(
            f"--depends-on {dep.variant}:{dep.seed}: upstream RUNNING but no "
            f"valid slurm.slurm_job_id tag — was it submitted via `graphids submit`?"
        )
    return ckpt, int(jid_tag)


def resolve_all(
    specs: list[DependencySpec], dataset: str
) -> tuple[list[tuple[str, str]], list[int]]:
    """Resolve every spec, returning ``(tla_pairs, afterok_jids)``.

    Each spec contributes one TLA (always) and at most one jid (only
    when the upstream is currently RUNNING). Variants not in
    :data:`DEPENDS_ON_TLA` raise :class:`typer.BadParameter`.
    """
    tlas: list[tuple[str, str]] = []
    jids: list[int] = []
    for spec in specs:
        if spec.variant not in DEPENDS_ON_TLA:
            raise typer.BadParameter(
                f"--depends-on {spec.variant}: not in dependency registry. "
                f"Register it in graphids/slurm/dependencies.py:DEPENDS_ON_TLA "
                f"or pass --tla manually. Known: {sorted(DEPENDS_ON_TLA)}"
            )
        ckpt, jid = resolve_dependency(spec, dataset)
        tlas.append((DEPENDS_ON_TLA[spec.variant], str(ckpt)))
        if jid is not None:
            jids.append(jid)
    return tlas, jids
