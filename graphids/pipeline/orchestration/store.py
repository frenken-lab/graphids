"""Pipeline state store with SQLite and PostgreSQL backends.

Three tables (inspired by Parsl's monitoring.db):
- job: job definition (UUID PK, parameters as JSON, parent UUIDs)
- attempt: per-execution try (native scheduler ID, timing, exit code)
- transition: append-only state log (job_id, state, timestamp)

All writes use transactions. The DB survives process crashes.
Resume = re-run the driver; it reads existing state and skips completed jobs.

Backend selection: pass a URI string to PipelineStore.
- ``sqlite:///path/to/db`` or a bare path → SQLite (default, zero deps)
- ``postgresql://user:pass@host:port/db`` → PostgreSQL (requires ``psycopg``)
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

_SCHEMA_SQLITE = """\
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

_SCHEMA_PG = """\
CREATE TABLE IF NOT EXISTS run (
    run_id    TEXT PRIMARY KEY,
    created   TIMESTAMPTZ NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS job (
    id         TEXT PRIMARY KEY,
    run_id     TEXT NOT NULL REFERENCES run(run_id),
    name       TEXT NOT NULL,
    executable TEXT NOT NULL DEFAULT '',
    arguments  JSONB NOT NULL DEFAULT '[]',
    parameters JSONB NOT NULL DEFAULT '{}',
    resources  JSONB NOT NULL DEFAULT '{}',
    parents    JSONB NOT NULL DEFAULT '[]',
    environment JSONB NOT NULL DEFAULT '{}',
    tags       JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS attempt (
    attempt_id  SERIAL PRIMARY KEY,
    job_id      TEXT NOT NULL REFERENCES job(id),
    native_id   TEXT,
    hostname    TEXT,
    submitted_at TIMESTAMPTZ,
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    exit_code   INTEGER,
    failure_reason TEXT,
    resources_used JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS transition (
    id        SERIAL PRIMARY KEY,
    job_id    TEXT NOT NULL REFERENCES job(id),
    state     TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    attempt_id INTEGER REFERENCES attempt(attempt_id),
    detail    TEXT
);

CREATE INDEX IF NOT EXISTS idx_job_run ON job(run_id);
CREATE INDEX IF NOT EXISTS idx_transition_job ON transition(job_id);
CREATE INDEX IF NOT EXISTS idx_attempt_job ON attempt(job_id);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_uri(uri: str | Path) -> tuple[str, str]:
    """Return (backend, connection_string). Bare paths become sqlite:///."""
    uri_str = str(uri)
    if uri_str.startswith("postgresql://") or uri_str.startswith("postgres://"):
        return "pg", uri_str
    if uri_str.startswith("sqlite:///"):
        # sqlite:///path → keep the absolute path (3rd slash is part of path)
        return "sqlite", uri_str[len("sqlite://") :]
    # Bare path — treat as SQLite
    return "sqlite", uri_str


class PipelineStore:
    """Dual-backend pipeline state store (SQLite or PostgreSQL)."""

    def __init__(self, uri: str | Path):
        self._backend, conn_str = _parse_uri(uri)

        if self._backend == "pg":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as e:
                raise ImportError(
                    "PostgreSQL backend requires psycopg. Install with: uv pip install -e '.[db]'"
                ) from e
            self._conn = psycopg.connect(conn_str, autocommit=False, row_factory=dict_row)
            with self._conn.cursor() as cur:
                cur.execute(_SCHEMA_PG)
            self._conn.commit()
        else:
            db_path = Path(conn_str)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path), isolation_level="DEFERRED")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA_SQLITE)

    @property
    def is_pg(self) -> bool:
        return self._backend == "pg"

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── SQL dialect helpers ──────────────────────────────────────────────

    def _json_extract(self, column: str, key: str) -> str:
        """Return SQL expression for extracting a JSON key as text."""
        if self.is_pg:
            return f"{column}->>'{key}'"
        return f"json_extract({column}, '$.{key}')"

    def _placeholder(self) -> str:
        return "%s" if self.is_pg else "?"

    def _execute(self, sql: str, params: tuple = ()) -> object:
        """Execute with correct placeholder style."""
        if self.is_pg:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur
        return self._conn.execute(sql, params)

    def _fetchone(self, sql: str, params: tuple = ()) -> dict | None:
        if self.is_pg:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def _fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.is_pg:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()  # type: ignore[return-value]
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def _commit(self) -> None:
        self._conn.commit()

    # ── Run management ──────────────────────────────────────────────────

    def create_run(self, run_id: str, metadata: dict | None = None) -> None:
        p = self._placeholder()
        sql = f"INSERT INTO run (run_id, created, metadata) VALUES ({p}, {p}, {p})"
        if self.is_pg:
            sql = f"""INSERT INTO run (run_id, created, metadata) VALUES ({p}, {p}, {p}) ON CONFLICT (run_id) DO NOTHING"""
        else:
            sql = f"INSERT OR IGNORE INTO run (run_id, created, metadata) VALUES ({p}, {p}, {p})"
        self._execute(sql, (run_id, _now(), json.dumps(metadata or {})))
        self._commit()

    # ── Job management ──────────────────────────────────────────────────

    def insert_jobs(self, run_id: str, jobs: list[JobSpec]) -> int:
        """Insert jobs that don't already exist. Returns count of new jobs inserted."""
        p = self._placeholder()
        existing = {
            row["id"] for row in self._fetchall(f"SELECT id FROM job WHERE run_id = {p}", (run_id,))
        }
        new_jobs = [j for j in jobs if str(j.id) not in existing]
        if not new_jobs:
            return 0

        for j in new_jobs:
            self._execute(
                f"INSERT INTO job (id, run_id, name, executable, arguments, parameters, resources, parents, environment, tags) VALUES ({p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p}, {p})",
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
            self._execute(
                f"INSERT INTO transition (job_id, state, timestamp) VALUES ({p}, {p}, {p})",
                (str(j.id), JobState.PENDING.value, _now()),
            )
        self._commit()
        return len(new_jobs)

    def get_job(self, job_id: str | UUID) -> dict | None:
        p = self._placeholder()
        return self._fetchone(f"SELECT * FROM job WHERE id = {p}", (str(job_id),))

    # ── State queries ───────────────────────────────────────────────────

    def current_state(self, job_id: str | UUID) -> JobState:
        """Get the most recent state for a job."""
        p = self._placeholder()
        row = self._fetchone(
            f"SELECT state FROM transition WHERE job_id = {p} ORDER BY id DESC LIMIT 1",
            (str(job_id),),
        )
        return JobState(row["state"]) if row else JobState.PENDING

    def current_states(self, run_id: str) -> dict[str, JobState]:
        """Get current state for all jobs in a run. Returns {job_id: state}."""
        p = self._placeholder()
        rows = self._fetchall(
            f"""
            SELECT j.id, t.state
            FROM job j
            JOIN transition t ON t.job_id = j.id
            WHERE j.run_id = {p}
              AND t.id = (SELECT MAX(t2.id) FROM transition t2 WHERE t2.job_id = j.id)
            """,
            (run_id,),
        )
        return {row["id"]: JobState(row["state"]) for row in rows}

    def jobs_in_state(self, run_id: str, *states: JobState) -> list[dict]:
        """Get all jobs in the given state(s) with their full info."""
        p = self._placeholder()
        state_values = [s.value for s in states]
        placeholders = ",".join(p for _ in state_values)
        rows = self._fetchall(
            f"""
            SELECT j.*
            FROM job j
            JOIN transition t ON t.job_id = j.id
            WHERE j.run_id = {p}
              AND t.id = (SELECT MAX(t2.id) FROM transition t2 WHERE t2.job_id = j.id)
              AND t.state IN ({placeholders})
            """,
            (run_id, *state_values),
        )
        return rows

    def jobs_by_parameter(self, run_id: str, key: str, value: str) -> list[dict]:
        """Query jobs by a parameter value."""
        p = self._placeholder()
        json_expr = self._json_extract("parameters", key)
        rows = self._fetchall(
            f"SELECT * FROM job WHERE run_id = {p} AND {json_expr} = {p}",
            (run_id, value),
        )
        return rows

    # ── State transitions ───────────────────────────────────────────────

    def transition(
        self,
        job_id: str | UUID,
        state: JobState,
        attempt_id: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Record a state transition."""
        p = self._placeholder()
        self._execute(
            f"INSERT INTO transition (job_id, state, timestamp, attempt_id, detail) VALUES ({p}, {p}, {p}, {p}, {p})",
            (str(job_id), state.value, _now(), attempt_id, detail),
        )
        self._commit()

    # ── Attempt management ──────────────────────────────────────────────

    def create_attempt(self, job_id: str | UUID, native_id: str | None = None) -> int:
        """Create a new execution attempt. Returns the attempt_id."""
        p = self._placeholder()
        if self.is_pg:
            row = self._fetchone(
                f"INSERT INTO attempt (job_id, native_id, submitted_at) VALUES ({p}, {p}, {p}) RETURNING attempt_id",
                (str(job_id), native_id, _now()),
            )
            self._commit()
            return row["attempt_id"]  # type: ignore[index]
        else:
            cur = self._conn.execute(
                "INSERT INTO attempt (job_id, native_id, submitted_at) VALUES (?, ?, ?)",
                (str(job_id), native_id, _now()),
            )
            self._conn.commit()
            return cur.lastrowid  # type: ignore[return-value]

    def update_attempt(self, attempt_id: int, **fields) -> None:
        """Update attempt fields (native_id, started_at, finished_at, exit_code, etc.)."""
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
        p = self._placeholder()
        set_clause = ", ".join(f"{k} = {p}" for k in updates)
        self._execute(
            f"UPDATE attempt SET {set_clause} WHERE attempt_id = {p}",
            (*updates.values(), attempt_id),
        )
        self._commit()

    def attempt_count(self, job_id: str | UUID) -> int:
        p = self._placeholder()
        row = self._fetchone(
            f"SELECT COUNT(*) as cnt FROM attempt WHERE job_id = {p}", (str(job_id),)
        )
        return row["cnt"]  # type: ignore[index]

    def latest_attempt(self, job_id: str | UUID) -> dict | None:
        p = self._placeholder()
        return self._fetchone(
            f"SELECT * FROM attempt WHERE job_id = {p} ORDER BY attempt_id DESC LIMIT 1",
            (str(job_id),),
        )

    # ── Dependency resolution ───────────────────────────────────────────

    def ready_jobs(self, run_id: str) -> list[dict]:
        """Find PENDING jobs whose parents are all COMPLETED."""
        states = self.current_states(run_id)
        pending = self.jobs_in_state(run_id, JobState.PENDING)

        ready = []
        for job in pending:
            parent_ids = (
                json.loads(job["parents"]) if isinstance(job["parents"], str) else job["parents"]
            )
            if not parent_ids:
                ready.append(job)
                continue
            if all(states.get(pid) == JobState.COMPLETED for pid in parent_ids):
                ready.append(job)
        return ready

    def has_failed_parents(self, job_id: str | UUID, run_id: str) -> bool:
        """Check if any parent of this job is in a terminal failure state."""
        job = self.get_job(job_id)
        if not job:
            return False
        parent_ids = (
            json.loads(job["parents"]) if isinstance(job["parents"], str) else job["parents"]
        )
        states = self.current_states(run_id)
        return any(
            states.get(pid) in (JobState.FAILED, JobState.ABANDONED, JobState.CANCELED)
            for pid in parent_ids
        )

    # ── Summary ─────────────────────────────────────────────────────────

    def summary(self, run_id: str) -> dict[str, int]:
        """Count jobs in each state."""
        states = self.current_states(run_id)
        counts: dict[str, int] = {}
        for state in states.values():
            counts[state.value] = counts.get(state.value, 0) + 1
        return counts

    def total_jobs(self, run_id: str) -> int:
        p = self._placeholder()
        row = self._fetchone(f"SELECT COUNT(*) as cnt FROM job WHERE run_id = {p}", (run_id,))
        return row["cnt"]  # type: ignore[index]
