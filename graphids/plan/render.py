"""Render a plan module into a validated :class:`Plan`.

One call site shared by ``graphids run`` (writes JSON) and
``graphids plans describe`` (preview-only). Threads ``plan_id`` +
``git_sha`` + ``plan_module`` onto every fit/test row, then validates.
"""

from __future__ import annotations

import fnmatch
import importlib
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from graphids.plan.rows import Plan


def mint_plan_id() -> str:
    """RFC 9562 UUIDv7 — 48-bit ms timestamp + 4-bit version + 74 bits random.

    Lex-sortable == temporally-sortable, so ``ls plan_*.json | sort`` and
    MLflow tag ranges over ``graphids.plan_id`` are temporally ordered.
    """
    ts_ms = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 64) & 0xFFF
    rand_b = rand & ((1 << 62) - 1)
    val = (ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{val:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def git_sha() -> str:
    """Short git SHA of the working tree, or ``"unknown"`` outside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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

    return Plan.model_validate(
        {
            "plan_id": plan_id,
            "plan_module": plan_module,
            "plan_args": {"dataset": dataset, "seed": seed},
            "created_at": created_at or datetime.now(UTC).isoformat(timespec="seconds"),
            "rows": rows,
        }
    )
