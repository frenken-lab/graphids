"""Pipeline state store (SQLite) for sweep pipeline resumption.

Three tables:
- job: job definition (UUID PK, parameters as JSON)
- attempt: per-execution try (timing, exit code)
- transition: append-only state log (job_id, state, timestamp)

All writes use WAL-mode transactions. The DB survives process crashes.
Resume = re-read existing state and skip completed jobs.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .job import JobSpec, JobState

log = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS run (
    run_id    TEXT PRIMARY KEY,
    created   TEXT NOT NULL,
    metadata  TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS job (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES run(run_id),
    name       TEXT NOT NULL,
    executable TEXT NOT NULL DEFAULT '',
    arguments  TEXT NOT NULL DEFAULT '[]',
    parameters TEXT NOT NULL DEFAULT '{}',
    resources  TEXT NOT NULL DEFAULT '{}',
    parents    TEXT NOT NULL DEFAULT '[]',
    environment TEXT NOT NULL DEFAULT '{}',
    tags       TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS attempt (
    attempt_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL REFERENCES job(id),
    native_id   TEXT,
    hostname    TEXT,
    submitted_at TEXT,
    started_at  TEXT,
    finished_at TEXT,
    exit_code   INTEGER,
    failure_reason TEXT,
    resources_used TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transition (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id    TEXT NOT NULL REFERENCES job(id),
    state     TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    attempt_id INTEGER REFERENCES attempt(attempt_id),
    detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_run ON job(run_id);
CREATE INDEX IF NOT EXISTS idx_transition_job ON transition(job_id);
CREATE INDEX IF NOT EXISTS idx_attempt_job ON attempt(job_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class PipelineStore:
    """SQLite-backed pipeline state store."""

    def __init__(self, uri: str | Path):
        db_path = str(uri).removeprefix("sqlite:///").removeprefix("sqlite://")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, isolation_level="DEFERRED")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── Run management ──────────────────────────────────────────────────

    def create_run(self, run_id: str, metadata: dict | None = None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO run (run_id, created, metadata) VALUES (?, ?, ?)",
            (run_id, _now(), json.dumps(metadata or {})),
        )
        self._conn.commit()

    # ── Job management ──────────────────────────────────────────────────

    def insert_jobs(self, run_id: str, jobs: list[JobSpec]) -> int:
        """Insert jobs that don't already exist. Returns count of new jobs inserted."""
        existing = {
            row["id"]
            for row in self._conn.execute("SELECT id FROM job WHERE run_id = ?", (run_id,))
        }
        new_jobs = [j for j in jobs if str(j.id) not in existing]
        for j in new_jobs:
            self._conn.execute(
                "INSERT INTO job (id, run_id, name, executable, arguments, parameters, resources, parents, environment, tags) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(j.id),
                    run_id,
                    j.name,
                    j.executable,
                    json.dumps(j.arguments),
                    json.dumps(j.parameters),
                    j.resources.model_dump_json(),
                    json.dumps([str(pid) for pid in j.parents]),
                    json.dumps(j.environment),
                    json.dumps(j.tags),
                ),
            )
            self._conn.execute(
                "INSERT INTO transition (job_id, state, timestamp) VALUES (?, ?, ?)",
                (str(j.id), JobState.PENDING.value, _now()),
            )
        self._conn.commit()
        return len(new_jobs)

    # ── State queries ───────────────────────────────────────────────────

    def current_state(self, job_id: str | UUID) -> JobState:
        """Get the most recent state for a job."""
        row = self._conn.execute(
            "SELECT state FROM transition WHERE job_id = ? ORDER BY id DESC LIMIT 1",
            (str(job_id),),
        ).fetchone()
        return JobState(row["state"]) if row else JobState.PENDING

    def jobs_by_parameter(self, run_id: str, key: str, value: str) -> list[dict]:
        """Query jobs by a parameter value."""
        rows = self._conn.execute(
            f"SELECT * FROM job WHERE run_id = ? AND json_extract(parameters, '$.{key}') = ?",
            (run_id, value),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── State transitions ───────────────────────────────────────────────

    def transition(
        self,
        job_id: str | UUID,
        state: JobState,
        attempt_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Record a state transition."""
        self._conn.execute(
            "INSERT INTO transition (job_id, state, timestamp, attempt_id, detail) VALUES (?, ?, ?, ?, ?)",
            (str(job_id), state.value, _now(), attempt_id, detail),
        )
        self._conn.commit()

    # ── Attempt management ──────────────────────────────────────────────

    def create_attempt(self, job_id: str | UUID, native_id: str | None = None) -> int:
        """Create a new execution attempt. Returns the attempt_id."""
        cur = self._conn.execute(
            "INSERT INTO attempt (job_id, native_id, submitted_at) VALUES (?, ?, ?)",
            (str(job_id), native_id, _now()),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def update_attempt(self, attempt_id: int, **fields) -> None:
        """Update attempt fields (finished_at, exit_code, failure_reason, etc.)."""
        allowed = {
            "native_id",
            "hostname",
            "started_at",
            "finished_at",
            "exit_code",
            "failure_reason",
            "resources_used",
        }
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        self._conn.execute(
            f"UPDATE attempt SET {set_clause} WHERE attempt_id = ?",
            (*updates.values(), attempt_id),
        )
        self._conn.commit()
