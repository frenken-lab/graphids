"""Render a plan module into a validated :class:`Plan`.

One call site shared by ``graphids run`` (writes JSON) and
``graphids plans describe`` (preview-only). Threads ``plan_id`` +
``git_sha`` + ``plan_module`` onto every fit/test row, then validates.
"""

from __future__ import annotations

import fnmatch
import importlib
from datetime import UTC, datetime

from graphids.plan.identity import git_sha, mint_plan_id
from graphids.plan.schema import Plan


def render_plan(
    plan_module: str,
    *,
    dataset: str,
    seed: int,
    filter_glob: str | None = None,
    created_at: str | None = None,
) -> Plan:
    """Import ``graphids.plan.plans.<plan_module>``, call ``build``, validate.

    ``filter_glob`` (fnmatch over row name) raises ``ValueError`` on zero
    matches with the available row names listed — same UX as the CLI's
    ``typer.BadParameter``, but framework-agnostic.
    """
    mod = importlib.import_module(f"graphids.plan.plans.{plan_module}")
    rows = mod.build(dataset=dataset, seed=seed)
    if filter_glob is not None:
        kept = [r for r in rows if fnmatch.fnmatchcase(r["name"], filter_glob)]
        if not kept:
            names = ", ".join(r["name"] for r in rows)
            raise ValueError(
                f"--filter {filter_glob!r} matched 0/{len(rows)} rows. Available: {names}"
            )
        rows = kept

    plan_id = mint_plan_id()
    sha = git_sha()
    for r in rows:
        r["plan_id"] = plan_id
        if r.get("action") in {"fit", "test"}:
            r["plan_module"] = plan_module
            r["git_sha"] = sha

    return Plan.model_validate({
        "plan_id": plan_id,
        "plan_module": plan_module,
        "plan_args": {"dataset": dataset, "seed": seed},
        "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": rows,
    })
