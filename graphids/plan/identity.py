"""Plan-identity helpers — `plan_id` minting and working-tree git SHA.

Both surface as MLflow tags via ``_mlflow.identity_tags``; together with
``plan_module``/``plan_args``/``row_name`` they encode the
reproduction contract documented in ``chassis-invariants.md``.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


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
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
